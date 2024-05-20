import re
import os
from tqdm import tqdm

def load_document_tokens(doc_token_dir):
    linespan_list, ancestor_list, doc_list = [], [], []
    with open(doc_token_dir) as file:
        for line in file:
            linespan, ancestor, doc = parse_line(line)
            if doc.startswith('[['):
                linespan_list.append(linespan)
                ancestor_list.append(ancestor)
                doc_list.append(doc.strip())
    return linespan_list, ancestor_list, doc_list

def parse_line(line):
    parts = line.split('\t')
    linespan = parts[0].strip('[').strip(']').split(', ')
    ancestor = parts[1].strip('[').strip(']').split(', ')
    doc = parts[2]
    return linespan, ancestor, doc

def extract_doc_tokens(doc_list):
    return [get_tokens_from_doc(doc) for doc in doc_list]

def extract_doc_tokens_type(doc_list):
    return [get_token_types(doc) for doc in doc_list]

def store_normalized_docs(linespan_list, ancestor_list, doc_tokens_norm, output_path, filename):
    sol = os.path.basename(filename).split('.')[0] + '_normalized.txt'
    with open(os.path.join(output_path, sol), 'w') as file:
        for e in zip(linespan_list, ancestor_list, doc_tokens_norm):
            file.write('_'.join(e[0]) + '\t' + ' '.join(e[1]) + '\t' + ' '.join(e[2]) + '\n')

def get_tokens_from_doc(statement_str):
    token_clean_exp = ' (@[0-9]+,[0-9]+:[0-9]+=) | (,<[0-9]+>,[0-9]+:[0-9]+) | (,<[0-9]+>,channel=[0-9],[0-9]+:[0-9]+) | (,<-1>,[0-9]+:[0-9]+)'
    token_clean = re.sub(token_clean_exp, '', statement_str)
    tokens = token_clean.strip().strip('[').strip(']').split('], [')
    return [e.strip("'") for e in tokens]

def get_token_types(statement_str):
    token_type_exp = ',<([0-9]+|-1)>,'
    return re.findall(token_type_exp, statement_str)

