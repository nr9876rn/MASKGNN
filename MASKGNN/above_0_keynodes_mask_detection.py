from torch_geometric.data import Data
import torch
import random
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
import os, sys
import numpy as np
import math
import logging
from collections import Counter
from transformers import BertModel, BertTokenizer

import hashlib

import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, global_max_pool, BatchNorm, JumpingKnowledge, global_mean_pool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from cfg_graph_detect_confige import get_config
from train_model.key_nodes_mask_GCN import GCN


def get_bert_features(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # 使用 CLS token 的特征


def hash_str(string, embedding_dim=64):
    # 使用MD5哈希函数
    hash_object = hashlib.md5(str(string).encode())
    # 获取哈希结果的字节串
    hash_digest = hash_object.digest()

    # 归一化操作，将整数值映射到[0, 1]范围
    normalized_values = np.array([
        int.from_bytes(hash_digest[i:i + 4], 'little') % embedding_dim / embedding_dim
        for i in range(0, 16, 4)
    ], dtype=np.float32)

    # 如果需要更高维度的嵌入，可以重复使用哈希值
    if embedding_dim > 4:
        repeats = (embedding_dim + 3) // 4  # 计算需要重复多少次
        normalized_values = np.tile(normalized_values, repeats)[:embedding_dim]

    return normalized_values


def extract_subgraph(cfg_graph, key_nodes):
    nodes, edges = cfg_graph

    # 提取关键节点ID列表
    key_node_ids = {node['idx'] for node in key_nodes}

    # 提取关键节点之间的边
    subgraph_edges = []
    for edge_key, edge_value in edges.items():
        src, dst, edge_info = edge_value
        if src in key_node_ids and dst in key_nodes:
            subgraph_edges.append((src, dst, edge_info))

    return subgraph_edges


# 将标签文本转换为张量
def label_to_tensor(label):
    label_mapping = {
        'normal': 0,  # 也用于 nan 值
        'Extra gas consumption': 1,
        'Integer overflow or underflow': 2,
        'Reentrancy': 3,
        'Timestamp dependence': 4
    }

    # 将 NaN 或空值视为正常合约
    label = 'normal' if pd.isna(label) else label
    return torch.tensor([label_mapping[label]], dtype=torch.long)


def save_data(all_data, file_path):
    torch.save(all_data, file_path)
    print(f"Data saved to {file_path}")


def load_data(path, all_data_path, min_nodes=3):
    # 检查是否已经有预处理好的数据文件
    if os.path.exists(all_data_path):
        print("Loading preprocessed data from file.")
        all_data = torch.load(all_data_path)
    else:
        print("Preprocessed data file not found. Processing data...")
        all_data = []
        for filename in tqdm(os.listdir(path), desc='Loading data'):
            if filename.endswith('.pt'):
                file_path = os.path.join(path, filename)
                data = torch.load(file_path)
                cfg_graph = data['cfg_graph']

                key_nodes = data['key_nodes']
                if len(key_nodes) < min_nodes:
                    continue

                graph_nodes = cfg_graph[0].values()
                graph_edges = extract_subgraph(cfg_graph, graph_nodes)
                label = data['contract_label']
                # 转换标签
                label_tensor = label_to_tensor(label)

                # 获取节点数量
                num_nodes = len(graph_nodes)

                node_features = [get_bert_features(node['Expr']['str']) for node in graph_nodes]
                node_features = torch.cat(node_features, dim=0)

                # 构建边索引
                if graph_edges:
                    edge_index = torch.tensor([[edge[0], edge[1]] for edge in graph_edges], dtype=torch.long).t().contiguous()
                else:
                    # 如果没有边，为每个节点添加自循环
                    edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)

                key_node_indices = [node['idx'] for node in key_nodes]  # 从key_nodes中提取所有索引
                key_node_mask = torch.zeros(len(graph_nodes), dtype=torch.bool)  # 创建一个全False的掩码
                key_node_mask[key_node_indices] = True  # 将关键节点位置设为True

                graph_data = Data(x=node_features, edge_index=edge_index, y=label_tensor, key_node_mask=key_node_mask)
                all_data.append(graph_data)

        save_data(all_data, all_data_path)
    return all_data


def standardize_features(x, mean, std):
    return (x - mean) / (std + 1e-5)


def balance_classes(data, labels, target_count=1000):
    label_to_data = {label: [] for label in set(labels)}
    for item, label in zip(data, labels):
        label_to_data[label].append(item)

    balanced_data = []
    balanced_labels = []

    for label, items in label_to_data.items():
        if len(items) > target_count:
            # 负采样
            sampled_items = random.sample(items, target_count)
        else:
            # 正采样
            sampled_items = items
            sampled_items += random.choices(items, k=target_count - len(items))

        balanced_data.extend(sampled_items)
        balanced_labels.extend([label] * target_count)

    return balanced_data, balanced_labels


def train_and_evaluate(all_data, num_epochs=300, patience=30):
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler('Log/filtered_model_log.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    config = get_config()

    for data in all_data:
        data.y[data.y == 4] = 3

    # all_data = [data for data in all_data if data.y in [0, 2]]

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用GPU
    else:
        device = torch.device("cpu")  # 如果GPU不可用，使用CPU

    current_random_state = np.random.randint(0, 10000)

    # 提取数据和标签
    labels = [data.y.item() for data in all_data]

    # 平衡各类别的数据
    balanced_data, balanced_labels = balance_classes(all_data, labels)

    # 使用 train_test_split 进行分层抽样
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        balanced_data, balanced_labels, test_size=0.2, stratify=balanced_labels, random_state=current_random_state)

    # train_data, valid_data, train_labels, valid_labels = train_test_split(
    #     all_data, labels, test_size=0.2, stratify=labels, random_state=32)

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["batch_size"], shuffle=False)

    # 输出训练和验证集类别分布检查
    train_labels = [data.y.item() for data in train_data]
    valid_labels = [data.y.item() for data in valid_data]

    train_counter = Counter(train_labels)
    valid_counter = Counter(valid_labels)

    print(f"Training set class distribution: {train_counter}")
    print(f"Validation set class distribution: {valid_counter}")

    # 计算训练数据的均值和标准差用于标准化
    feature_stack = torch.cat([data.x for data in train_loader], dim=0)
    train_mean = feature_stack.mean(dim=0)
    train_std = feature_stack.std(dim=0)

    # 初始化模型、损失函数和优化器
    model = GCN(num_features=config["num_features"], hidden_channels=config["hidden_channels"], num_classes=config["num_classes"])
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],  weight_decay=config["weight_decay"])
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

    # 用于早停的追踪
    best_val_loss = float('inf')
    best_val_train_loss = float('inf')
    epochs_without_improvement = 0

    # 用于保存最佳指标
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_f1 = 0

    print(
        f'Parameters: num_epochs={num_epochs}, patience={patience}, num_features={config["num_features"]}, batch_size={config["batch_size"]}'
        f'hidden_channels={config["hidden_channels"]}, num_classes={config["num_classes"]}, lr={config["lr"]}, weight_decay={config["weight_decay"]}')
    # Log the parameters
    logger.info(f'Parameters: num_epochs={num_epochs}, patience={patience}, num_features={config["num_features"]}, batch_size={config["batch_size"]}'
                f'hidden_channels={config["hidden_channels"]}, num_classes={config["num_classes"]}, lr={config["lr"]}, weight_decay={config["weight_decay"]}')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            # 标准化特征
            data.x = standardize_features(data.x, train_mean, train_std)
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        # scheduler.step()  # 更新学习率

        # 验证阶段
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in valid_loader:
                # 标准化特征
                data.x = standardize_features(data.x, train_mean, train_std)
                data = data.to(device)

                output = model(data)
                loss = criterion(output, data.y)
                total_val_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(data.y.tolist())

        avg_val_loss = total_val_loss / len(valid_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)


        # Log the results
        logger.info(
            f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n'
            # f'Confusion Matrix:\n{cm}'
        )

        # 打印训练和验证结果
        print(
            f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n'
            # f'Confusion Matrix:\n{cm}'
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

        # 更新最佳指标并检查早停条件
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_train_loss = avg_train_loss
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                # print('Early stopping triggered.')
                break


    # Log the best results
    logger.info(f'Best Accuracy: {best_accuracy:.4f}')
    logger.info(f'Best Precision: {best_precision:.4f}')
    logger.info(f'Best Recall: {best_recall:.4f}')
    logger.info(f'Best F1 Score: {best_f1:.4f}')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    logger.info(f'Best validation_train loss: {best_val_train_loss:.4f}')

    print(f'Best Accuracy: {best_accuracy:.4f}')
    print(f'Best Precision: {best_precision:.4f}')
    print(f'Best Recall: {best_recall:.4f}')
    print(f'Best F1 Score: {best_f1:.4f}')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation_train loss: {best_val_train_loss:.4f}')

    best_results={
        'Best Accuracy': best_accuracy,
        'Best Precision': best_precision,
        'Best F1 Score': best_f1,
        'random state': current_random_state,
    }

    return best_results


if __name__ == '__main__':
    # 加载 BertTiny 模型和 tokenizer
    model_directory = r'E:\Academicfiles\pycharmfiles\Key-AST SC\path_to_model\bert-tiny'
    # 加载本地模型
    tokenizer = BertTokenizer.from_pretrained(model_directory)
    model = BertModel.from_pretrained(model_directory)

    # 路径定义
    # path = r'E:\Academicfiles\pycharmfiles\Key-AST SC\data\key_v_above_0_keynodes'
    # all_data_path = 'data/cfg_data_128d.pth'

    path = r'E:/Academicfiles/pycharmfiles/Key-AST SC/data/ablation_top3v'
    all_data_path = 'data/ablation_origin_128d.pth'

    all_data = load_data(path, all_data_path)
    train_and_evaluate(all_data)

    # while True:
    #     result = train_and_evaluate(all_data)
    #     if result['Best Accuracy'] > 0.88:
    #         print(result)
    #     if result['Best Accuracy'] >= 0.91:
    #         break