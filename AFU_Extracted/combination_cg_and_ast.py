import networkx as nx
import os
import pickle
import re
from tqdm import tqdm

def print_nx_network_full_info(nx_graph):
    print('====Nodes info====')
    for node, node_data in nx_graph.nodes(data=True):
        print(node, node_data)
    
    print('====Edges info====')
    for source_node, target_node, edge_data in nx_graph.edges(data=True):
        print(source_node, target_node, edge_data)

def preorder_traversal(graph, source_node, ast_data, output_directory, file_prefix):
    visited = set()  
    stack = [source_node] 

    matched_ast = []  

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            node_info = graph.nodes[node]
            code_lines = node_info.get('node_source_code_lines', [None, None])

          
            for line_range in range(code_lines[0], code_lines[1]+1):
                for line in ast_data.split('\n'):
                    if f'{line_range}_' in line:
                        matched_ast.append(line)

         
            for _, child in graph.out_edges(node):
                if child not in visited:
                    stack.append(child)
    

    if matched_ast:
        with open(os.path.join(output_directory, f'{file_prefix}_ast_preorder.txt'), 'a') as f:
            f.write('\n'.join(matched_ast) + '\n\n') 

def process_and_match_ast(cg_dir, ast_dir, output_dir):
    for filename in tqdm(os.listdir(cg_dir)):
        if filename.endswith(".pkl"):
            # Load call graph
            with open(os.path.join(cg_dir, filename), 'rb') as f:
                cg_data = pickle.load(f)
                graph = nx.MultiDiGraph(cg_data)

            # Load AST file
            ast_filename = filename.split('.')[0] + '_parse_result_normalized'
            with open(os.path.join(ast_dir, ast_filename), 'r') as f:
                ast_data = f.read()

            # Find a source node to start preorder traversal, assuming graph is a tree
            # This could be the first node with no incoming edges, or root nodes based on domain knowledge
            for node in graph.nodes:
                if graph.in_degree(node) == 0:  # Found a root node
                    preorder_traversal(graph, node, ast_data, output_dir, filename.split('.')[0])
                    break  # Remove this break if you want to start traversal from all roots

if __name__ == "__main__":
    cg_dir = './AFU_Extracted/output/cg/'
    ast_dir = './AFU_Extracted/output/norm/'
    output_dir = './AFU_Extracted/output/AFU/'
    
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process and match AST with call graph
    process_and_match_ast(cg_dir, ast_dir, output_dir)
