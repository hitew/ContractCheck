import re
import os
from tqdm import tqdm

def normalize_doc_tokens(doc_tokens, doc_tokens_type):
    return [normalize_tokens(tokens, token_type) for tokens, token_type in zip(doc_tokens, doc_tokens_type)]

def normalize_tokens(token_list, token_type):
    token_list_normalized = []
    for i, token in enumerate(token_list):
        if token_type[i] in ['117', '118', '2', '15', '34']:
            continue
        elif token_type[i] == '95':
            token = "VersionLiteral"
        elif token_type[i] in ['115', '97', '98', '100']:
            token = "normalizedToken"
        token_list_normalized.append(token.lower() if not token.startswith('%%%%') else token)
    return token_list_normalized
    
def normalize_files(base_dir, output_dir):
    for addr in tqdm(os.listdir(base_dir)):
        if addr.endswith(".txt"):  # Assuming the documents are .txt files
            try:
                doc_token_dir = os.path.join(base_dir, addr)
                linespan_list, ancestor_list, doc_list = load_document_tokens(doc_token_dir)
                doc_tokens = extract_doc_tokens(doc_list)
                doc_tokens_type = extract_doc_tokens_type(doc_list)
                doc_tokens_norm = normalize_doc_tokens(doc_tokens, doc_tokens_type)
                store_normalized_docs(linespan_list, ancestor_list, doc_tokens_norm, output_dir, addr)
                print("Success:", addr)
            except Exception as e:
                print(f"Error processing {addr}: {e}")

if __name__ == "__main__":
    base_dir = './AST_Parsed/output/'
    output_dir = './AFU_Extracted/output/norm/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_files(base_dir, output_dir)
