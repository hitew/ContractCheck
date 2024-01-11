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





# with open("/home/wangxite/hitework/TEST/0x2019763bd984cce011cd9b55b0e700abe42fa6c7.sol.pkl", "rb") as f:
#     graph = pickle.load(f)



cg_dir = './AFU_Extracted/output/cg/'
ast_dir = './AFU_Extracted/output/norm/'
output_dir = './AFU_Extracted/output/AFU/'

def process_and_match_ast(cg_dir, ast_dir, output_dir):
    for filename in tqdm(os.listdir(cg_dir)):
        if filename.endswith(".pkl"):
           
            with open(os.path.join(cg_dir, filename), 'rb') as f:
                cg_data = pickle.load(f)
                graph = nx.MultiDiGraph(cg_data)

            
            ast_filename = filename.split('.')[0] + 'parse_result_normalized'
            with open(os.path.join(ast_dir, ast_filename), 'r') as f:
                ast_data = f.read()

            with open(os.path.join(output_dir, f'{filename}_ast.txt'), 'a') as f:
                f.write(ast_data + '\n')
            for source_node, target_node, edge_data in graph.edges(data=True):
             
                source_node_info = graph.nodes[source_node]
                target_node_info = graph.nodes[target_node]

              
                source_code_lines = source_node_info['node_source_code_lines']
                target_code_lines = target_node_info['node_source_code_lines']

                
                matched_ast = []
                for line_range in range(source_code_lines[0], source_code_lines[1]+1):
                    for line in ast_data.split('\n'):
                        if f'{line_range}_' in line:
                            matched_ast.append(line)

                for line_range in range(target_code_lines[0], target_code_lines[1]+1):
                    for line in ast_data.split('\n'):
                        if f'{line_range}_' in line:
                            matched_ast.append(line)

                if matched_ast:
                  
                    
                    
                    # Write matched_ast information
                    with open(os.path.join(output_dir, f'{filename}_ast.txt'), 'a') as f:
                        f.write('\n'.join(matched_ast) + '\n')

process_and_match_ast(cg_dir, ast_dir, output_dir)