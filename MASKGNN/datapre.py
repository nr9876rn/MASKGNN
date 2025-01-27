import os
import torch
import pandas as pd
import re
import subprocess
from slither.slither import Slither
from slither.core.cfg.node import NodeType
from collections import defaultdict
from tqdm import tqdm

def get_solidity_version(path):
    pragma_line = ""
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if 'pragma solidity' in line and not re.match(r'\s*//', line.strip()):
                pragma_line = line
                break
    # 正则表达式匹配版本号
    version_match = re.search(r'^\s*pragma\s+solidity\s*[\^>=]*\s*(\d+\.\d+\.\d+)\s*;', pragma_line)
    # version_match = re.search(r'\^(\d+\.\d+\.\d+)', pragma_line)
    return version_match.group(1) if version_match else None

# def set_solidity_version(requested_version):
#     # 已安装的Solidity版本，假设每个版本可以向下兼容同一主版本号的较低版本
#     installed_versions = ["0.4.25", "0.5.5", "0.6.6", "0.7.4", "0.8.0"]
#
#     major_version_requested = int(requested_version.split('.')[1])
#     compatible_version = None
#
#     for version in installed_versions:
#         major_version_installed = int(version.split('.')[1])
#         if major_version_installed == major_version_requested:
#             compatible_version = version
#             break
#     if compatible_version is None:
#         print(f"No compatible version found for requested version {requested_version}")
#         raise ValueError("No compatible installed version found")
#     subprocess.run(
#         ["solc-select", "use", version],
#         check=True,
#         stdout=subprocess.DEVNULL,  # Suppress standard output
#         stderr=subprocess.DEVNULL  # Suppress standard error
#     )

def set_solidity_version(version):
    """设置Solidity编译器版本."""
    try:
        subprocess.run(
                ["solc-select", "use", version],
                check=True,
                stdout=subprocess.DEVNULL,  # Suppress standard output
                stderr=subprocess.DEVNULL  # Suppress standard error
            )
    except subprocess.CalledProcessError as e:
        print(f"Failed to set Solidity version {version}: {str(e)}")


def install_compiler(versions):
    for version in versions:
        command = f'solc-select install {version}'
        subprocess.run(command, shell=True, check=True)


'''找到特定后缀的文件'''
def FindFile(fileDirname, fileEXT):
    # filepath     = 'D:\\srcdata\\Gaelic_test_1.mp3'
    # fileDirname  = 'D:\\srcdata'
    # foldername   = 'srcdata'
    # fileBasename = 'Gaelic_test_1.mp3'
    # fileTitle    = 'Gaelic_test_1'
    # fileEXT      = '.mp3'
    # fileType     = 'mp3'
    #
    # os.path.dirname(filepath)      == fileDirname
    # os.path.basename(filepath)     == fileBasename
    # os.path.split(filepath)        == (fileDirname, fileBasename)
    # os.path.split(fileDirname)     == (fileDirname, foldername)
    # os.path.splitext(filepath)     == ('D:\\srcdata\\Gaelic_test_1', '.mp3')
    # os.path.splitext(fileBasename) == (fileTitle, fileEXT)

    filepaths = []
    fileBasenames = []
    fileTitles = []
    for fileBasename in os.listdir(fileDirname):
        filepath = os.path.join(fileDirname, fileBasename)
        if os.path.splitext(fileBasename)[1] == fileEXT:  # 判断文件类型
            filepaths.append(os.path.join(filepath))  # 文件路径
            fileBasenames.append(fileBasename)  # 文件名（带后缀）
            fileTitles.append(os.path.splitext(fileBasename)[0])  # 文件标题
    return filepaths, fileBasenames, fileTitles


# 加载所有 .pt 文件
def load_pt_files(folder):
    data_list = []
    for filename in os.listdir(folder):
        if filename.endswith('.pt'):
            data_path = os.path.join(folder, filename)
            data = torch.load(data_path)
            data_list.append((filename, data))
    return data_list


# 获取 `funcLevel_label_slither` 中的高风险函数
def get_high_risk_functions(func_level_label_slither):
    high_risk_functions = set()
    for entry in func_level_label_slither:
        high_risk_functions.update(entry['High Risk Functions'].split(','))
    return high_risk_functions


# 根据 Solidity 源代码路径创建 AST
def create_ast_from_solidity(sourcecode_path):
    full_path = os.path.join(contract_folder, sourcecode_path)
    slither = None

    # processed = False
    versions = [
         "0.4.24", "0.4.25", "0.4.10", "0.4.11", "0.4.12", "0.4.13", "0.4.14", "0.4.15", "0.4.16", "0.4.17", "0.4.18", "0.4.19", "0.4.20",
        "0.4.21", "0.4.22", "0.4.23", "0.4.26", "0.5.0", "0.5.1", "0.5.2", "0.5.3", "0.5.4",
        "0.5.5", "0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10", "0.5.11", "0.5.12", "0.5.13", "0.5.14", "0.5.15",
        "0.5.16", "0.5.17", "0.6.0", "0.6.1", "0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8", "0.6.9",
        "0.6.10", "0.6.11", "0.6.12", "0.7.0", "0.7.1", "0.7.2", "0.7.3", "0.7.4", "0.7.5", "0.7.6", "0.8.0", "0.8.1",
        "0.8.2", "0.8.3", "0.8.4", "0.8.5", "0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10", "0.8.11", "0.8.12", "0.8.13",
        "0.8.14", "0.8.15", "0.8.16", "0.8.17", "0.8.18", "0.8.19", "0.8.20", "0.8.21", "0.8.22", "0.8.23"
    ]

    for version in versions:
        set_solidity_version(version)
        try:
            slither = Slither(full_path)
            break
        except Exception:
            continue

    if not slither:
        print(f"Failed to create Slither instance for any version for file: {sourcecode_path}")
    return slither

# 获取 AST 节点所在的函数名称
def get_function_name_from_node(node):
    func = node.function
    if func:
        return func.name
    return None

# 遍历 AST 并建立节点与函数的映射关系
def traverse_ast_and_map_nodes(slither, high_risk_functions):
    node_to_function = defaultdict(lambda: {'vuln_count': 0, 'non_vuln_count': 0})

    for contract in slither.contracts:
        # 获取合约中的所有函数
        for func in contract.functions_and_modifiers:
            is_vuln_function = func.name in high_risk_functions
            for node in func.nodes:
                # 处理变量读取节点
                for var in node.variables_read:
                    if var and var.name:
                        if is_vuln_function:
                            node_to_function[var.name]['vuln_count'] += 1
                        else:
                            node_to_function[var.name]['non_vuln_count'] += 1
                # 处理内部和外部调用节点
                if node.expression:
                    expression_str = str(node.expression)
                    if is_vuln_function:
                        node_to_function[expression_str]['vuln_count'] += 1
                    else:
                        node_to_function[expression_str]['non_vuln_count'] += 1

    return node_to_function


# 创建 AST-function 表
def create_ast_function_table(data_list):
    records = []
    for filename, data in data_list:
        sourcecode_info = data['sourcecode_path']
        sourcecode_path = sourcecode_info['sourcecode_path']
        slither = create_ast_from_solidity(sourcecode_path)

        high_risk_functions = get_high_risk_functions(data['funcLevel_label_slither'])

        node_to_function = traverse_ast_and_map_nodes(slither, high_risk_functions)

        for var_name, counts in node_to_function.items():
            records.append([
                # filename,
                var_name,
                counts['vuln_count'],
                counts['non_vuln_count'],
                data['contLevel_label']['vulnerability'],
                ','.join(high_risk_functions)
            ])

    return records

def create_single_ast_function_table(data):
    records = []
    # for filename, data in data_list:
    sourcecode_info = data['sourcecode_path']
    sourcecode_path = sourcecode_info['sourcecode_path']
    slither = create_ast_from_solidity(sourcecode_path)
    if slither:
        high_risk_functions = get_high_risk_functions(data['funcLevel_label_slither'])

        node_to_function = traverse_ast_and_map_nodes(slither, high_risk_functions)

        for var_name, counts in node_to_function.items():
            records.append([
                sourcecode_path,
                var_name,
                counts['vuln_count'],
                counts['non_vuln_count'],
                data['contLevel_label']['vulnerability_type'],
                ','.join(high_risk_functions)
            ])

        # print(records)
        return records
    else:
        print(print(f"Failed to process {sourcecode_path} with any version"))
        return None

if __name__ == '__main__':
    # print("111111111111")
    # 文件路径配置
    pt_folder = r'E:\Project\SmtCon_dataset\classmodel\minidataset\Tsinghua\processed'
    contract_folder = r'E:\Project\SmtCon_dataset\contractcode'
    output_file = 'AST_function_mapping.csv'

    versions = [
        "0.4.24", "0.4.25", "0.4.10", "0.4.11", "0.4.12", "0.4.13", "0.4.14", "0.4.15", "0.4.16", "0.4.17", "0.4.18", "0.4.19", "0.4.20",
        "0.4.21", "0.4.22", "0.4.23", "0.4.26", "0.5.0", "0.5.1", "0.5.2", "0.5.3", "0.5.4",
        "0.5.5", "0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10", "0.5.11", "0.5.12", "0.5.13", "0.5.14", "0.5.15",
        "0.5.16", "0.5.17", "0.6.0", "0.6.1", "0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8", "0.6.9",
        "0.6.10", "0.6.11", "0.6.12", "0.7.0", "0.7.1", "0.7.2", "0.7.3", "0.7.4", "0.7.5", "0.7.6", "0.8.0", "0.8.1",
        "0.8.2", "0.8.3", "0.8.4", "0.8.5", "0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10", "0.8.11", "0.8.12", "0.8.13",
        "0.8.14", "0.8.15", "0.8.16", "0.8.17", "0.8.18", "0.8.19", "0.8.20", "0.8.21", "0.8.22", "0.8.23"
    ]

    # install_compiler(versions)

    filepaths, fileBasenames, fileTitles = FindFile(pt_folder, ".pt")

    # data_list = []
    ast_function_records = []

    # # 主程序逻辑
    # data_list = load_pt_files(pt_folder)

    # 为每个数据对象创建 AST 并存为 'AST_GPT'
    for filepath in tqdm(filepaths, desc="Processing files"):
        # print("222222222222")
        data = torch.load(filepath)
        sourcecode_info = data['sourcecode_path']
        sourcecode_path = sourcecode_info['sourcecode_path']
        # slither = create_ast_from_solidity(sourcecode_path)
        # print("333333333333333")
        # data['AST_GPT'] = slither
        # data_list.append(data)
        records = create_single_ast_function_table(data)
        for record in records:
            # print(record)
            ast_function_records.append(record)
            print(ast_function_records[-1])


    # 构建 AST-function 表
    # ast_function_records = create_ast_function_table(data_list)

    # 将表格保存为 CSV 文件
    columns = ['contract_filename', 'variable_name', 'vuln_function_count', 'non_vuln_function_count',
               'contract_vuln_label', 'vuln_functions']
    df = pd.DataFrame(ast_function_records, columns=columns)
    df.to_csv(output_file, index=False)

    print(f'AST-function mapping has been saved to {output_file}')
