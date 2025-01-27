import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
else:
    device = torch.device("cpu")  # 如果GPU不可用，使用CPU


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        key_node_mask = data.key_node_mask
        x, edge_index, batch = self.filter_data(x, edge_index, batch, key_node_mask)

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def filter_data(self, x, edge_index, batch, key_node_mask):
        # Apply the key_node_mask to x
        x = x[key_node_mask]

        # Create a mapping from old indices to new indices
        new_index = torch.zeros_like(key_node_mask, dtype=torch.long)
        new_index[key_node_mask] = torch.arange(key_node_mask.sum(), device=device)

        # Filter edges to keep only those connecting key nodes
        edge_index = edge_index[:, key_node_mask[edge_index[0]] & key_node_mask[edge_index[1]]]
        edge_index = new_index[edge_index]

        # Filter the batch tensor to match the new x
        batch = batch[key_node_mask]

        # Check if edge_index is empty and add self-loops to key nodes
        if edge_index.size(1) == 0:
            # Create self-loops for the filtered nodes
            num_nodes = key_node_mask.sum()
            self_loops = torch.arange(num_nodes, device=edge_index.device)
            self_loops = self_loops.unsqueeze(0).repeat(2, 1)
            edge_index = self_loops

        return x, edge_index, batch

