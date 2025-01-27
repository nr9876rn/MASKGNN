import re
import subprocess
import os
import torch
from tqdm import tqdm
from solc import compile_files  # 注意: 不要安装solc, 而是安装py-solc！！！！！！！！！！！！！！！！
from slither import Slither  # 注意: 不要安装slither, 而是安装slither-analyzer！！！！！！！！！！！！！！！！
import networkx as nx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
from slither.core.cfg.node import NodeType
from slither.core.declarations.function import FunctionType
from slither.core import expressions
# from dataprep.IR.solc_AST.explore_AST import parse_contract_versions, set_compiler, get_installed_versions
# from dataprep.IR.Slither_CFG_sourcecode.CFG_sourcecode_V2 import CFG_sourcecode_generator_expression, draw_graph

def get_installed_versions():
	# 调用命令行获取已安装的版本
	result = subprocess.run(["solc-select", "versions"], capture_output=True, text=True)
	installed_versions = re.findall(r'\d+\.\d+\.\d+', result.stdout)
	installed_versions.sort()
	return installed_versions

def set_compiler(version):
	command = f'solc-select use {version}'
	subprocess.run(command, stdout=subprocess.DEVNULL, shell=True, check=True)

def CFG_sourcecode_generator_expression(contract_file_path):
	slither = None
	versions = [
		"0.4.24", "0.4.25", "0.4.10", "0.4.11", "0.4.12", "0.4.13", "0.4.14", "0.4.15", "0.4.16", "0.4.17", "0.4.18",
		"0.4.19", "0.4.20",
		"0.4.21", "0.4.22", "0.4.23", "0.4.26", "0.5.0", "0.5.1", "0.5.2", "0.5.3", "0.5.4",
		"0.5.5", "0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10", "0.5.11", "0.5.12", "0.5.13", "0.5.14", "0.5.15",
		"0.5.16", "0.5.17", "0.6.0", "0.6.1", "0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8", "0.6.9",
		"0.6.10", "0.6.11", "0.6.12", "0.7.0", "0.7.1", "0.7.2", "0.7.3", "0.7.4", "0.7.5", "0.7.6", "0.8.0", "0.8.1",
		"0.8.2", "0.8.3", "0.8.4", "0.8.5", "0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10", "0.8.11", "0.8.12", "0.8.13",
		"0.8.14", "0.8.15", "0.8.16", "0.8.17", "0.8.18", "0.8.19", "0.8.20", "0.8.21", "0.8.22", "0.8.23"
	]
	for version in versions:
		set_compiler(version)
		try:
			slither = Slither(contract_file_path)
			break
		except Exception:
			continue

	merge_contract_graph = None
	statementNode_list = []
	expressionNode_list = []
	for contract in slither.contracts:
		merged_graph = None
		for function in contract.functions + contract.modifiers:

			nx_g = nx.MultiDiGraph()
			for node in function.nodes:
				statementNode = StatementNode(node)
				if statementNode not in statementNode_list:
					statementNode_list.append(statementNode)
				for expnode in statementNode.set_expnode:
					if expnode not in expressionNode_list:
						expressionNode_list.append(expnode)
						nx_g.add_node(expnode)

			nx_graph = nx_g
			if merged_graph is None:
				merged_graph = nx_graph.copy()
			else:
				merged_graph = nx.compose(merged_graph, nx_graph)

		if merge_contract_graph is None:
			if merged_graph is not None:
				merge_contract_graph = merged_graph.copy()
			else:
				merge_contract_graph = None
		elif merged_graph is not None:
			merge_contract_graph = nx.compose(merge_contract_graph, merged_graph)

	'''
		此时，图的所有节点已经构建完成，接下来构建边, 分3步：
	'''
	# 1. 遍历所有的StatementNode，找到StatementNode之间的边
	for statementNode in statementNode_list:
		current_node = statementNode.statementnode
		# 检查statementNode中的边情况
		if current_node.type in [NodeType.IF, NodeType.IFLOOP]:
			son_true_node = current_node.son_true
			if son_true_node:
				son_true_statementNode_list = [node for node in statementNode_list if node.statementnode == son_true_node]
				assert len(son_true_statementNode_list) <= 1, "找到多个son_true_statementNode"
				if len(son_true_statementNode_list) == 1:
					son_true_statementNode = son_true_statementNode_list[0]
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_true_statementNode.set_expnode[0], edge_type='if_true')
			son_false_node = current_node.son_false
			if son_false_node:
				son_false_statementNode_list = [node for node in statementNode_list if node.statementnode == son_false_node]
				assert len(son_false_statementNode_list) <= 1, "找到多个son_true_statementNode"
				if len(son_false_statementNode_list) == 1:
					son_false_statementNode = son_false_statementNode_list[0]
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_false_statementNode.set_expnode[0], edge_type='if_false')
		else:
			for son_node in current_node.sons:
				if son_node:
					# 找到statementNode_list中statementNode.statementnode == son_node的StatementNode
					son_statementNode_list = [node for node in statementNode_list if node.statementnode == son_node]
					if len(son_statementNode_list) == 0:
						continue
					son_statementNode = son_statementNode_list[0]
					# 构建一条边
					merge_contract_graph.add_edge(statementNode.set_expnode[-1], son_statementNode.set_expnode[0], edge_type='next')

	# 2. 遍历所有的StatementNode，找到StatementNode之内的ExpressionNode的边
	for statementNode in statementNode_list:
		# statementNode.set_expnode是一个列表，只需要将列表从头到尾连起来
		for i in range(len(statementNode.set_expnode) - 1):
			merge_contract_graph.add_edge(statementNode.set_expnode[i], statementNode.set_expnode[i + 1], edge_type='next')

	# 3. 遍历所有的ExpressionNode，找到CallExpression和SuperCallExpression
	# 	根据它们的called，找到对应的ExpressionNode，构建边
	for expnode_idx, expnode in enumerate(expressionNode_list):
		if expnode.exprs_type in ["CallExpression", "SuperCallExpression"]:

			# 如果调用的函数不是Identifier，则跳过此调用，主要避免一些特殊情况
			if str(expnode.exprs.called.__class__) != "<class 'slither.core.expressions.identifier.Identifier'>":
				continue

			FunctionCall = expnode.exprs.called.value

			# 找到这个Function的初始statement的第一个ExpressionNode，保存在first_expnode_of_func_list中，作为后续Callnode的出边，详细步骤：
			# 	1. 找到所有属于此Function的初始statement，即确保statement id为0
			# 	2. 从statement中找到第一个ExpressionNode
			first_expnode_of_func_list = []
			for son_expnode in expressionNode_list:
				if son_expnode.function == FunctionCall and son_expnode.state_id == 0 and son_expnode.contract_uniqID == expnode.contract_uniqID:
					# 从son_expnode找到对应的statementNode，将statementNode的第一个ExpressionNode添加到new_son_expnode_list中
					first_node = son_expnode.statementnode.set_expnode[0]
					first_expnode_of_func_list.append(first_node)
			# 如果调用的函数中没有ExpressionNode，则跳过此调用
			if len(first_expnode_of_func_list) == 0:
				continue
			assert len(first_expnode_of_func_list) == 1, "调用的function的初始表达式数量不为1, 而是{0}".format(len(first_expnode_of_func_list))
			first_expnode_of_func = first_expnode_of_func_list[0]

			# 找到这个Function的最后statement的最后一个ExpressionNode，保存在last_expnode_of_func_list中，作为后续Sonnode的入边，详细步骤：
			# 	1. 找到此Function的所有statement，并确保statement id为最大
			# 	2. 从statement中找到最后一个ExpressionNode
			last_expnode_of_func_list = []
			statment_max_id = max([node.state_id for node in statementNode_list if node.function == FunctionCall and node.contract_uniqID == expnode.contract_uniqID])
			for son_expnode in expressionNode_list:
				if son_expnode.function == FunctionCall and son_expnode.state_id == statment_max_id and son_expnode.contract_uniqID == expnode.contract_uniqID:
					# 从son_expnode找到对应的statementNode，将statementNode的最后一个ExpressionNode添加到new_son_expnode_list中
					last_node = son_expnode.statementnode.set_expnode[-1]
					last_expnode_of_func_list.append(last_node)
			last_expnode_of_func_list = list(set(last_expnode_of_func_list))
			assert len(last_expnode_of_func_list) == 1, "调用的function的末尾表达式数量不为1, 而是{0}".format(len(last_expnode_of_func_list))
			last_expnode_of_func = last_expnode_of_func_list[0]

			# 找到与当前expnode存在边关系的orig_son_expnode, 同时orig_son_expnode得是出边，将它们的边删除
			orig_son_of_expnode_list = []
			edges = list(merge_contract_graph.edges(data=True))
			for edge in edges:
				if edge[0] == expnode:
					son_of_expnode = edge[1]
					attr = edge[2]
					orig_son_of_expnode_list.append(son_of_expnode)
					merge_contract_graph.remove_edge(expnode, son_of_expnode)

			# 重建边，有两类边：
			# 	1. expnode -> first_expnode_of_func_list，edge_type='call'
			# 	2. last_expnode_of_func -> orig_son_of_expnode，edge_type='callback'
			merge_contract_graph.add_edge(expnode, first_expnode_of_func, edge_type='call', last_expnode_of_func=last_expnode_of_func, orig_son_of_expnode_list=orig_son_of_expnode_list)
			for orig_son_of_expnode in orig_son_of_expnode_list:
				merge_contract_graph.add_edge(last_expnode_of_func, orig_son_of_expnode, edge_type='callback', first_expnode_of_func=first_expnode_of_func, expnode=expnode)

	return merge_contract_graph

Funct_Type = [
	"NORMAL",
	"CONSTRUCTOR",
	"FALLBACK",
	"RECEIVE",
	"CONSTRUCTOR_VARIABLES",  # Fake function to hold variable declaration statements
	"CONSTRUCTOR_CONSTANT_VARIABLES",
]

State_Type = [
	"ENTRY_POINT",  # no expression

	# Nodes that may have an expression
	"EXPRESSION",  # normal case
	"RETURN",  # RETURN may contain an expression
	"IF",
	"NEW VARIABLE",  # Variable declaration
	"INLINE ASM",
	"IF_LOOP",

	# Nodes where control flow merges
	# Can have phi IR operation
	"END_IF",  # ENDIF node source mapping points to the if/else "body"
	"BEGIN_LOOP",  # STARTLOOP node source mapping points to the entire loop "body"
	"END_LOOP",  # ENDLOOP node source mapping points to the entire loop "body"

	# Below the nodes do not have an expression but are used to expression CFG structure.

	# Absorbing node
	"THROW",

	# Loop related nodes
	"BREAK",
	"CONTINUE",

	# Only modifier node
	"_",

	"TRY",
	"CATCH",

	# Node not related to the CFG
	# Use for state variable declaration
	"OTHER_ENTRYPOINT",
	"END INLINE ASM"
]

Exprs_Type = [
	"AssignmentOperation",
	"BinaryOperation",
	"CallExpression",
	"ConditionalExpression",
	"ElementaryTypeNameExpression",
	'Identifier',
	"IndexAccess",
	"Literal",
	"MemberAccess",
	"NewArray",
	"NewContract",
	"NewElementaryType",
	"SuperCallExpression",
	"SuperIdentifier",
	"TupleExpression",
	"TypeConversion",
	"UnaryOperation",
	'NoneType'
]

class ExpressionNode:
	def __init__(self, statementnode, exprs, exp_id):
		# 下面是StatementNode的属性
		self.statementnode = statementnode
		# 所属的函数
		self.function = statementnode.function
		self.function_type = self.function.function_type
		# 所属的合约
		self.contract = statementnode.function.contract
		self.contract_kind = self.contract.contract_kind
		# statement的全称
		self.state_str = statementnode.state_str
		self.state_id = statementnode.state_id
		# 接下来是ExpressionNode的属性
		self.exprs = exprs
		self.exp_id = exp_id
		self.exprs_type = type(exprs).__name__
		self.exprs_str = exprs.__str__()

		# 所属的合约的唯一标识
		self.contract_uniqID = statementnode.contract_uniqID
		# 所属的函数的标识和唯一标识
		self.function_ID = statementnode.function_ID
		self.function_uniqID = statementnode.function_uniqID
		# statement的标识和唯一标识
		self.statement_ID = statementnode.statement_ID
		self.statement_uniqID = statementnode.statement_uniqID
		# expression的标识和唯一标识
		self.expression_ID = "Expr: " + str(self.exp_id) + "--" + self.exprs_type + "--" + self.exprs_str
		self.expression_uniqID = self.statement_uniqID + "__" + self.expression_ID

		# 构建expression
		self.expressions_inside()

	def expressions_inside(self):
		# 根据slither.core.expressions.Expression的类型，给出不同的处理当前节点中包含的expression的方式
		'''
		AssignmentOperation: 			expression_left: expression, expression_right: expression
		BinaryOperation: 				expression_left: expression, expression_right: expression
		CallExpression: 				called: expression, arguments: List[expression]
		ConditionalExpression: 			if_expression: expression, then_expression: expression, else_expression: expression
		ElementaryTypeNameExpression: 	None
		Identifier: 					None
		IndexAccess: 					expression_left: expression, expression_right: expression
		Literal: 						None
		MemberAccess: 					expression: expression
		NewArray: 						None
		NewContract: 					None
		NewElementaryType: 				None
		SuperCallExpression: 			called: expression, arguments: List[expression]
		SuperIdentifier: 				None
		TupleExpression: 				expressions: List[expression]
		TypeConversion: 				expression: expression
		UnaryOperation: 				expression: expression
		'''
		if isinstance(self.exprs, expressions.AssignmentOperation)\
				or isinstance(self.exprs, expressions.BinaryOperation)\
				or isinstance(self.exprs, expressions.IndexAccess):
			self.inside_exprs_list = [self.exprs.expression_right, self.exprs.expression_left]
		elif isinstance(self.exprs, expressions.CallExpression)\
				or isinstance(self.exprs, expressions.SuperCallExpression):
			self.inside_exprs_list = self.exprs.arguments + [self.exprs.called]
		elif isinstance(self.exprs, expressions.ConditionalExpression):
			self.inside_exprs_list = [self.exprs.else_expression, self.exprs.then_expression, self.exprs.if_expression]
		elif isinstance(self.exprs, expressions.MemberAccess):
			self.inside_exprs_list = [self.exprs.expression]
		elif isinstance(self.exprs, expressions.TupleExpression):
			self.inside_exprs_list = self.exprs.expressions
		elif isinstance(self.exprs, expressions.TypeConversion)\
				or isinstance(self.exprs, expressions.UnaryOperation):
			self.inside_exprs_list = [self.exprs.expression]
		else:
			self.inside_exprs_list = []

	def __repr__(self):
		return f"ExpressionNode({self.expression_uniqID})"

	def __hash__(self):
		return hash(self.expression_uniqID)

	def __eq__(self, other):
		return isinstance(other, ExpressionNode) and self.expression_uniqID == other.expression_uniqID

	def is_same_statement(self, other):
		return isinstance(other, ExpressionNode) and self.statement_uniqID == other.statement_uniqID

class StatementNode:
	def __init__(self, statementnode):
		self.statementnode = statementnode
		# 所属的函数
		self.function = statementnode.function
		self.function_type = self.function.function_type
		# 所属的合约
		self.contract = statementnode.function.contract
		self.contract_kind = self.contract.contract_kind
		# statement的全称
		self.state_str = statementnode.__str__()
		self.state_id = statementnode.node_id

		# 所属的合约的唯一标识
		self.contract_uniqID = "Cont: " + str(self.contract_kind) + "--" + self.contract.__str__()
		# 所属的函数的标识和唯一标识
		self.function_ID = "Func: " + self.function_type.name + "--" + self.function.__str__()
		self.function_uniqID = self.contract_uniqID + "__" + self.function_ID
		# statement的标识和唯一标识
		self.statement_ID = "Stat: " + str(self.state_id) + "--" + self.state_str
		self.statement_uniqID = self.function_uniqID + "__" + self.statement_ID

		# 从StatementNode中找到所有的expression，并构建ExpressionNode
		self.set_expnode = []
		self.exp_id = 0  # 用于标记expression的顺序, 从而唯一标识位于同一个statement的expression
		if self.statementnode.expression:
			self.contruct_expnode(self.statementnode.expression)
		# 如果set_expnode没有expression node，则set_expnode中加入一个None的ExpressionNode
		if not self.set_expnode:
			self.exp_id += 1
			self.set_expnode.append(ExpressionNode(self, None, self.exp_id))

	def contruct_expnode(self, exprs):
		self.exp_id += 1
		exp_id = self.exp_id
		exprs_type = type(exprs).__name__
		# 根据exprs，构建ExpressionNode
		exprs_node = ExpressionNode(self, exprs, exp_id)
		# 根据exprs_type，检查其中是否有expression, 如果有expression，则递归调用contruct_expnode
		if exprs_node.inside_exprs_list:
			for inside_exprs in exprs_node.inside_exprs_list:
				self.contruct_expnode(inside_exprs)
		# 将除了几个特别的expression外的其他的expression添加到set_expnode中
		if exprs_type not in ["Identifier", "Literal", "SuperIdentifier"]:
			self.set_expnode.append(exprs_node)
		# 如果没有expression，则递归结束，在返回的过程中，从最深一层的expression开始，
		# 每一层的ExpressionNode构建一条指向上一层的ExpressionNode的边
		pass

	def __repr__(self):
		return f"StatementNode(statement: {self.state_str}, function: {self.function.__str__()}, contract: {self.contract.__str__()})"

	def __hash__(self):
		return hash(self.statement_uniqID)

	def __eq__(self, other):
		return isinstance(other, StatementNode) and self.statement_uniqID == other.statement_uniqID

	def is_same_function(self, other):
		return isinstance(other, StatementNode) and self.function_uniqID == other.function_uniqID

	def is_same_contract(self, other):
		return isinstance(other, StatementNode) and self.contract_uniqID == other.contract_uniqID

class normGraph:
	def __init__(self, origGraph):
		'''
		节点属性：Function, Statement, State_Type, Exprs_Type
		边属性：出点，入点，边类型，(若为Call边)last_expnode_of_func和orig_son_of_expnode_list，(若为Callback边)first_expnode_of_func和expnode
		'''
		self.nxGraph = origGraph
		self.nxGnodes = list(origGraph.nodes)
		self.nxGedges = list(origGraph.edges(data=True))

		self.Funct_Type = Funct_Type
		self.State_Type = State_Type
		self.Exprs_Type = Exprs_Type

		self.node_dict = {}
		self.node_generate()

		self.edge_dict = {}
		self.edge_generate()

		self.Node_match_Edge()

	def node_generate(self):
		for idx, node in enumerate(self.nxGnodes):
			node_feat = {}
			node_feat["idx"] = idx
			node_feat["name"] = str(node)

			node_feat["Cont"] = {}
			node_feat["Cont"]["str"] = str(node.contract)
			node_feat["Cont"]["ID"] = node.contract_uniqID

			node_feat["Func"] = {}
			node_feat["Func"]["str"] = str(node.function)
			node_feat["Func"]["ID"] = node.function_ID
			node_feat["Func"]["Funct_Type"] = node.function_type.name
			node_feat["Func"]["Funct_Type_index"] = self.Funct_Type.index(node_feat["Func"]["Funct_Type"])

			node_feat["Stat"] = {}
			node_feat["Stat"]["str"] = node.state_str
			node_feat["Stat"]["state_sequenceNumb"] = node.state_id
			node_feat["Stat"]["State_Type"] = node.statementnode.statementnode.type.value
			node_feat["Stat"]["State_Type_index"] = self.State_Type.index(node_feat["Stat"]["State_Type"])

			node_feat["Expr"] = {}
			node_feat["Expr"]["str"] = node.exprs_str
			node_feat["Expr"]["Exprs_Type"] = node.exprs_type
			node_feat["Expr"]["Exprs_Type_index"] = self.Exprs_Type.index(node_feat["Expr"]["Exprs_Type"])

			self.node_dict[idx] = node_feat

	def edge_generate(self):
		# 先把self.nxGedges中的节点换成node_dict中的index，将结果存入self.edge_list
		edges = self.nxGedges
		for index_edge, edge in enumerate(edges):
			edge_feat = edge[2]
			# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[0]的idx
			outNode_idx = -1
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[0]):
					outNode_idx = node["idx"]
					break
			inNode_idx = -1
			# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[1]的idx
			for idx, node in self.node_dict.items():
				if node["name"] == str(edge[1]):
					inNode_idx = node["idx"]
					break

			if edge_feat["edge_type"] == "call":
				# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[2]["last_expnode_of_func"]的idx
				expnode = edge[2]["last_expnode_of_func"]
				last_expnode_of_func = None
				for idx, node in self.node_dict.items():
					if node["name"] == str(expnode):
						last_expnode_of_func = node["idx"]
						break
				# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[2]["orig_son_of_expnode_list"]的idx
				orig_son_of_expnode_list = []
				for expnode in edge[2]["orig_son_of_expnode_list"]:
					orig_son_of_expnode = None
					for idx, node in self.node_dict.items():
						if node["name"] == str(expnode):
							orig_son_of_expnode = node["idx"]
							break
					orig_son_of_expnode_list.append(orig_son_of_expnode)

				edge_feat["last_expnode_of_func"] = last_expnode_of_func
				edge_feat["orig_son_of_expnode_list"] = orig_son_of_expnode_list
			elif edge_feat["edge_type"] == "callback":
				# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[2]["first_expnode_of_func"]的idx
				first_expnode_of_func = -1
				for idx, node in self.node_dict.items():
					if node["name"] == str(edge[2]["first_expnode_of_func"]):
						first_expnode_of_func = node["idx"]
						break
				# 找到self.node_dict中self.node_dict[idx]["ID"] == edge[2]["expnode"]的idx
				expnode = -1
				for idx, node in self.node_dict.items():
					if node["name"] == str(edge[2]["expnode"]):
						expnode = node["idx"]
						break

				edge_feat["first_expnode_of_func"] = first_expnode_of_func
				edge_feat["expnode"] = expnode

			New_edge = (outNode_idx, inNode_idx, edge_feat)
			self.edge_dict[index_edge] = New_edge

		# 将self.edge_list中的call边和callback边进行绑定
		for index_edge in self.edge_dict.keys():
			edge = self.edge_dict[index_edge]
			outNode_idx = edge[0]
			inNode_idx = edge[1]
			edge_feat = edge[2]
			edge_type = edge_feat["edge_type"]

			if edge_type != "call":
				continue

			callback_outNode_idx = edge_feat["last_expnode_of_func"]
			callback_inNode_idx_list = edge_feat["orig_son_of_expnode_list"]

			callback_edge_index_list = []
			for callback_inNode_idx in callback_inNode_idx_list:
				for idx in self.edge_dict.keys():
					edge_callback = self.edge_dict[idx]
					if edge_callback[0] == callback_outNode_idx and edge_callback[1] == callback_inNode_idx and edge_callback[2]["edge_type"] == "callback":
						callback_edge_index_list.append(idx)
						self.edge_dict[idx][2]["call_edge_index"] = index_edge
						# break

			self.edge_dict[index_edge][2]["callback_edge_index_list"] = callback_edge_index_list

			pass

	def Node_match_Edge(self):
		# 遍历所有的节点，找到每个节点的以该节点为起点的所有边，并将这些边附加到node的字典中
		for idx in range(len(self.node_dict)):
			edge_list = []
			self.node_dict[idx]["edge_list"] = edge_list

		for edge in self.edge_dict.values():
			node_idx = edge[0]
			self.node_dict[node_idx]["edge_list"].append(edge)

		# # 打印node的edge情况
		# for node_idx in range(len(self.node_dict)):
		# 	edge_list = self.node_dict[node_idx]["edge_list"]
		# 	print("Node: ", node_idx)
		# 	for edge in edge_list:
		# 		print("\tIn: {}, Out: {}, type: {}".format(edge[0], edge[1], edge[2]["edge_type"]))


'''找到特定后缀的文件'''
def FindFile(fileDirname, fileEXT):
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



if __name__ == '__main__':
	# processed = False
	versions = [
		"0.4.24", "0.4.25", "0.4.10", "0.4.11", "0.4.12", "0.4.13", "0.4.14", "0.4.15", "0.4.16", "0.4.17", "0.4.18",
		"0.4.19", "0.4.20",
		"0.4.21", "0.4.22", "0.4.23", "0.4.26", "0.5.0", "0.5.1", "0.5.2", "0.5.3", "0.5.4",
		"0.5.5", "0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10", "0.5.11", "0.5.12", "0.5.13", "0.5.14", "0.5.15",
		"0.5.16", "0.5.17", "0.6.0", "0.6.1", "0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8", "0.6.9",
		"0.6.10", "0.6.11", "0.6.12", "0.7.0", "0.7.1", "0.7.2", "0.7.3", "0.7.4", "0.7.5", "0.7.6", "0.8.0", "0.8.1",
		"0.8.2", "0.8.3", "0.8.4", "0.8.5", "0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10", "0.8.11", "0.8.12", "0.8.13",
		"0.8.14", "0.8.15", "0.8.16", "0.8.17", "0.8.18", "0.8.19", "0.8.20", "0.8.21", "0.8.22", "0.8.23"
	]

	pt_folder = r'E:\Project\SmtCon_dataset\classmodel\minidataset\Tsinghua\processed'
	contract_folder = r'E:\Project\SmtCon_dataset\SmartContract\Tsinghua\contractcode'
	output_dir = r'E:\Academicfiles\pycharmfiles\Key-AST SC\data\constructed_cfg'

	filepaths, fileBasenames, fileTitles = FindFile(pt_folder, ".pt")

	for filepath in tqdm(filepaths, desc="Processing files"):
		data = torch.load(filepath)
		sourcecode_info = data['sourcecode_path']
		sourcecode_path = sourcecode_info['sourcecode_path']
		contract_sol_path = os.path.join(contract_folder, sourcecode_path)

		full_graph = CFG_sourcecode_generator_expression(contract_sol_path)
		if not full_graph:
			print(filepath)
			continue

		nG = normGraph(full_graph)
		SCcfg = (nG.node_dict, nG.edge_dict)

		result = {
			'filename': sourcecode_path,
			'cfg_graph': SCcfg,
			'contract_label': data['contLevel_label']['vulnerability_type']
		}

		output_file = os.path.join(output_dir, f"{sourcecode_path}_CFG.pt")  # 构建文件名

		# 保存数据（确保result中的数据可以转换为Tensor）
		torch.save(result, output_file)