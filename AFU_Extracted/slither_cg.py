import os
import re
import networkx as nx
import subprocess
from copy import deepcopy
from os.path import join
from scipy.integrate._ivp.radau import C
from slither.slither import Slither
from collections import defaultdict
from networkx.algorithms import cluster
from slither.core.cfg.node import Node, NodeType
from tqdm import tqdm
import logging 
from slither.printers.call import call_graph
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.function import Function
from slither.core.variables.variable import Variable
import pickle
logger = logging.getLogger("Slither-simil") 


pattern =  re.compile(r'\d.\d.\d+')
def get_solc_version(source):
    with open(source, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return '0.4.25'
            line = f.readline()
    return '0.4.25'


# return graph edge with edge type
def _edge(from_node, to_node, edge_type, label):
    return (from_node, to_node, edge_type, label)
    

# return unique id for contract function to use as node name
def _function_node(contract, function, filename_input):
    source_mapping = str(function.source_mapping)

    line_range = source_mapping.split('#')[1].split('-')
    #print(line_range)
    if len(line_range)==1:
        start_line = int(line_range[0])
        end_line = int(line_range[0])
    else:
        start_line = int(line_range[0])
        end_line = int(line_range[1])
    

    node_info = {
        'node_id': f"{filename_input}_{contract.id}_{contract.name}_{function.full_name}",
        'label': f"{filename_input}_{contract.name}_{function.full_name}",
        'function_fullname': function.full_name, 
        'contract_name': contract.name, 
        'source_file': filename_input,
        'node_source_code_lines': (start_line, end_line)
    }

    return node_info

# return unique id for solidity function to use as node name
def _solidity_function_node(solidity_function):
    # node_function_source_code_lines = solidity_function.source_mapping['lines']
    node_info = {
        'node_id': f"[Solidity]_{solidity_function.full_name}",
        'label': f"[Solidity]_{solidity_function.full_name}",
        'function_fullname': solidity_function.full_name,
        'contract_name': None,
        'source_file': None,
        'node_source_code_lines': None
    }
    # return f"[Solidity]_{solidity_function.full_name}"
    return node_info

# return node info from a node tupple
def _get_node_info(tuple_node):
    if tuple_node[0][0] == 'node_id':
        node_id = tuple_node[0][1]
    if tuple_node[1][0] == 'label':
        node_label = tuple_node[1][1]
    if tuple_node[2][0] == 'function_fullname':
        function_fullname = tuple_node[2][1]
    if tuple_node[3][0] == 'contract_name':
        contract_name = tuple_node[3][1]
    if tuple_node[4][0] == 'source_file':
        source_file = tuple_node[4][1]
    if tuple_node[5][0] == 'node_source_code_lines':
        node_function_source_code_lines = list(tuple_node[5][1])
    

    if 'fallback' in node_id:
        node_type = 'fallback_function'
    elif '[Solidity]' in node_id:
        node_type = 'fallback_function'
    else:
        node_type = 'contract_function'
    
    return node_id, node_label, node_type, function_fullname, contract_name, source_file, node_function_source_code_lines

# return edge info from a contract call tuple
def _add_edge_info_to_nxgraph(contract_call, nx_graph):
    source = contract_call[0]
    source_node_id, source_label, source_type, source_function_fullname, source_contract_name, \
    source_source_file, source_node_function_source_code_lines = _get_node_info(source)

    if source_node_id not in nx_graph.nodes():
        nx_graph.add_node(source_node_id, label=source_label, node_type=source_type,
                          node_source_code_lines=source_node_function_source_code_lines,
                          function_fullname=source_function_fullname, contract_name=source_contract_name,
                          source_file=source_source_file)

    target = contract_call[1]
    target_node_id, target_label, target_type, target_function_fullname, target_contract_name, \
    target_source_file, target_node_function_source_code_lines = _get_node_info(target)

    if target_node_id not in nx_graph.nodes():
        nx_graph.add_node(target_node_id, label=target_label, node_type=target_type,
                          node_source_code_lines=target_node_function_source_code_lines,
                          function_fullname=target_function_fullname, contract_name=target_contract_name,
                          source_file=target_source_file)

    edge_type = contract_call[2]
    edge_label = contract_call[3]

    nx_graph.add_edge(source_node_id, target_node_id, label=edge_label, edge_type=edge_type)

# pylint: disable=too-many-arguments
def _process_internal_call(
    contract,
    function,
    internal_call,
    contract_calls,
    solidity_functions,
    solidity_calls,
    filename_input,
):

    if isinstance(internal_call, (Function)):
        # print('tuple:', tuple(_function_node(contract, function, filename_input).items()))
        contract_calls[contract].add(
            _edge(

                tuple(_function_node(contract, function, filename_input).items()),
                tuple(_function_node(contract, internal_call, filename_input).items()),
                edge_type='internal_call',
                label='internal_call'
            )
        )

    elif isinstance(internal_call, (SolidityFunction)):
        solidity_functions.add(tuple(_solidity_function_node(internal_call).items()))
        solidity_calls.add(
            _edge(
                tuple(_function_node(contract, function, filename_input).items()),
                tuple(_solidity_function_node(internal_call).items()),
                edge_type='solidity_call',
                label='solidity_call'
            )
        )

def _process_external_call(
    contract,
    function,
    external_call,
    contract_functions,
    external_calls,
    all_contracts,
    filename_input,

):

    external_contract, external_function = external_call
    
    if not external_contract in all_contracts:
        return

    # add variable as node to respective contract
    if isinstance(external_function, (Variable)):
        contract_functions[external_contract].add(tuple(
                _function_node(external_contract, external_function, filename_input).items()))


    external_calls.add(
        _edge(
            tuple(_function_node(contract, function, filename_input).items()),
            tuple(_function_node(external_contract, external_function, filename_input).items()),
            edge_type='external_call',
            label='external_call'
        )
    )

def _render_internal_calls(nx_graph, contract, contract_functions, contract_calls):
    if len(contract_functions[contract]) > 0:
        for contract_function in contract_functions[contract]:     
            node_id, node_label, node_type, function_fullname, contract_name, source_file, \
            node_function_source_code_lines = _get_node_info(contract_function)

            nx_graph.add_node(node_id, label=node_label, node_type=node_type,

                              node_source_code_lines=node_function_source_code_lines,
                              function_fullname=function_fullname, contract_name=contract_name,
                              source_file=source_file)
    
    if len(contract_calls[contract]) > 0:
        for contract_call in contract_calls[contract]:
            _add_edge_info_to_nxgraph(contract_call, nx_graph)


def _render_solidity_calls(nx_graph, solidity_functions, solidity_calls):
    if len(solidity_functions) > 0:
        for solidity_function in solidity_functions:
            # print(solidity_function)
            node_id, node_label, node_type, function_fullname, contract_name, source_file, \
             node_function_source_code_lines = _get_node_info(solidity_function)

            nx_graph.add_node(node_id, label=node_label, node_type=node_type,
                              node_source_code_lines=node_function_source_code_lines,
                              function_fullname=function_fullname, contract_name=contract_name,
                              source_file=source_file)
    
    if len(solidity_calls) > 0:
        for solidity_call in solidity_calls:
            _add_edge_info_to_nxgraph(solidity_call, nx_graph)

def _render_external_calls(nx_graph, external_calls):
    if len(external_calls) > 0:
        for external_call in external_calls:
            _add_edge_info_to_nxgraph(external_call, nx_graph)

def _process_function(
    contract,
    function,
    contract_functions,
    contract_calls,
    solidity_functions,
    solidity_calls,
    external_calls,
    all_contracts,
    filename_input,

):  

    contract_functions[contract].add(tuple(
        _function_node(contract, function, filename_input).items())
    )

    for internal_call in function.internal_calls:
        _process_internal_call(
            contract,
            function,
            internal_call,
            contract_calls,
            solidity_functions,
            solidity_calls,
            filename_input
        )

    for external_call in function.high_level_calls:
        _process_external_call(
            contract,
            function,
            external_call,
            contract_functions,
            external_calls,
            all_contracts,
            filename_input

        )

def _process_functions(functions, filename_input):
    contract_functions = defaultdict(set)  # contract -> contract functions nodes
    contract_calls = defaultdict(set)  # contract -> contract calls edges

    solidity_functions = set()  # solidity function nodes
    solidity_calls = set()  # solidity calls edges
    external_calls = set()  # external calls edges

    all_contracts = set()

    for function in functions:
        all_contracts.add(function.contract_declarer)

    for function in functions:
        _process_function(
            function.contract_declarer,
            function,
            contract_functions,
            contract_calls,
            solidity_functions,
            solidity_calls,
            external_calls,
            all_contracts,
            filename_input
        )

    # print('contract_functions:', contract_functions)
    # print('solidity_functions:', solidity_functions)
    # print('contract_calls:', contract_calls)
    # print('solidity_calls:', solidity_calls)
    # print('external_calls:', external_calls)
    # print('all_contracts:', all_contracts)

    all_contracts_graph = nx.MultiDiGraph()
    for contract in all_contracts:
        _render_internal_calls(all_contracts_graph, contract,
                               contract_functions, contract_calls)
    
    # _render_solidity_calls(all_contracts_graph, solidity_functions, solidity_calls)
    _render_external_calls(all_contracts_graph, external_calls)

    return all_contracts_graph




def extract_graph(source_path, output,logger):
    sc_version = get_solc_version(source_path)
   
    solc_select_cmd = f"solc-select use {sc_version}"
    process1 = subprocess.Popen(solc_select_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    slither = Slither(source_path)
    print(slither)
    

    file_name_sc = source_path.split('/')[-1]
    call_graph_printer = GESCPrinters(slither, logger, file_name_sc)
    all_contracts_call_graph = call_graph_printer.generate_all_contracts_call_graph()  
    file_path = os.path.join(output, file_name_sc + ".pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(all_contracts_call_graph, f)
        
    

    with open(file_path, 'rb') as f:
       data = pickle.load(f)

    print(data)
    return 1



class GESCPrinters(AbstractPrinter):
    ARGUMENT = 'call-graph'
    HELP = 'Export the call-graph of the contracts to a dot file and a gpickle file'

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph'
    def __init__(self, slither, logger, filename):
        super().__init__(slither,logger)
        self.filename = filename
        

    def generate_all_contracts_call_graph(self):
        # Avoid dupplicate funcitons due to different compilation unit
        all_functionss = [
            compilation_unit.functions for compilation_unit in self.slither.compilation_units
        ]
        all_functions = [item for sublist in all_functionss for item in sublist]
        all_functions_as_dict = {
            function.canonical_name: function for function in all_functions
        }

        all_contracts_call_graph = _process_functions(all_functions_as_dict.values(), self.filename)

        return all_contracts_call_graph

    def output(self):
        """
        Output the graph in filename
        Args:
            filename(string)
        """





def extract_all_graphs(source_dir, output_dir, logger):
    for file_name in tqdm(os.listdir(source_dir)):
        if file_name.endswith(".sol"):
            source_path = os.path.join(source_dir, file_name)
            try:
                extract_graph(source_path, output_dir, logger)
            except:
                continue

source_dir = "./data/vun_data/"
output_dir = "./AFU_Extracted/output/cg/"
extract_all_graphs(source_dir, output_dir, logger)
