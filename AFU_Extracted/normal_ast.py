
import re
import os
from tqdm import tqdm

class Statement_Normalizer(object):

    def __init__(self, doc_token_dir):

        self.doc_token_dir = doc_token_dir
        self.linespan_list, \
        self.ancestor_list, \
        self.doc_list = self.load_document_tokens()
        self.doc_tokens = self.extract_doc_tokens()
        self.doc_tokens_type = self.extract_doc_tokens_type()
        self.doc_tokens_norm = self.normalize_doc_tokens()

        assert len(self.linespan_list) == len(self.ancestor_list)
        assert len(self.ancestor_list) == len(self.doc_list)
        assert len(self.doc_list) == len(self.doc_tokens_norm)
        assert len(self.doc_tokens) == len(self.doc_tokens_type)

    def load_document_tokens(self):
        linespan_list = []
        ancestor_list = []
        doc_list = []
        with open(self.doc_token_dir ) as dt:
            for i, line in enumerate(dt):
                linespan = line.split('\t')[0].strip('[').strip(']').split(', ')
                ancestor = line.split('\t')[1].strip('[').strip(']').split(', ')
                doc = line.split('\t')[2]
                if doc.startswith('[['):
                    linespan_list.append(linespan)
                    ancestor_list.append(ancestor)
                    doc_list.append(doc.strip())
                else:
                    continue
        return linespan_list, ancestor_list, doc_list

    def extract_doc_tokens(self):
        doc_tokens = []
        for doc in self.doc_list:
            tokens = self.get_tokens_from_doc(doc)
            doc_tokens.append(tokens)
        return doc_tokens

    def extract_doc_tokens_type(self):
        doc_tokens_type = []
        for doc_str in self.doc_list:
            tokens_type = self.get_token_types(doc_str)
            doc_tokens_type.append(tokens_type)
        return doc_tokens_type

    def normalize_doc_tokens(self):
        doc_tokens_norm = []
        for i in range(len(self.doc_list)):
            token = self.doc_tokens[i]
            token_type = self.doc_tokens_type[i]
            assert len(token) == len(token_type)
            tokens_norm = self.normalize_tokens(token, token_type)
            doc_tokens_norm.append(tokens_norm)
        return doc_tokens_norm

    def store(self):
        store_path = './AFU_Extracted/output/norm/'
        sol = os.path.basename(self.doc_token_dir)
        with open(store_path + sol + '_normalized', 'w') as sdt:
            for e in zip(self.linespan_list, self.ancestor_list, self.doc_tokens_norm):
                sdt.write('_'.join(e[0]) + '\t')
                sdt.write(' '.join(e[1]) + ' ')
                sdt.write(' '.join(e[2]) + '\n')

    @staticmethod
    def get_tokens_from_doc(statement_str):
        token_clean_exp = '@[0-9]+,[0-9]+:[0-9]+=' + '|' \
                                                     ',<[0-9]+>,[0-9]+:[0-9]+' + '|',<[0-9]+>,channel=[0-9],[0-9]+:[0-9]+' + '|',<-1>,[0-9]+:[0-9]+'
        token_clean_reg = re.compile(token_clean_exp)
        token_clean = re.sub(token_clean_reg, '', statement_str)
        tokens = token_clean.strip().strip('[').strip(']').split('], [')
        return [e.strip("'") for e in tokens]

    @staticmethod
    def get_token_types(statement_str):
        token_type_exp = ',<[0-9]+>,|,<-1>,'
        token_type_reg = re.compile(token_type_exp)
        token_type = token_type_reg.findall(statement_str)
        return token_type

    @staticmethod
    def normalize_tokens(token_list, token_type):
        assert len(token_list) == len(token_type)
        token_list_normalized = Statement_Normalizer.execute_token_normalization(token_list, token_type)
        return [token.lower() if not token.startswith('%%%%') else token for token in token_list_normalized]

    @staticmethod
    def execute_token_normalization(token_list, token_type):
        assert len(token_list) == len(token_type)
        token_list_normalized = []
        for i in range(len(token_list)):
            token_type_spec = token_type[i]
            if token_type_spec in [",<117>,", ",<118>,", ",<2>,", ",<15>,", ",<34>,"]:
                continue
            elif token_type_spec == ',<95>,':
                token_list[i] = "VersionLiteral"
            elif token_type_spec in [",<115>,", ",<97>,", ",<98>,", ",<100>,"]:
                token_list[i] = "normalizedToken"
            elif token_type_spec == ",<114>,":
                token_list_normalized.extend(Statement_Normalizer.handle_identifier(token_list[i]))
            else:
                token_list_normalized.append(token_list[i])
        return token_list_normalized

    @staticmethod
    def handle_identifier(identifier):
        token_list_normalized = ['%%%%' + identifier]
        camel_case_exp = '([A-Z][a-z]+)'
        camel_case_reg = re.compile(camel_case_exp)
        camel_case_cut = camel_case_reg.findall(identifier)
        if not camel_case_cut:
            token_list_normalized.append(identifier)
        else:
            remain = identifier
            for e in camel_case_cut:
                remain = remain.replace(e, '')
            if remain:
                token_list_normalized.append(remain)
            token_list_normalized.extend(camel_case_cut)
            token_list_normalized.extend([i for i in identifier.split('_') if i])
        return token_list_normalized


base_dir = './AST_Parsed/output/'
contract_addrs = os.listdir(base_dir)

for addr in tqdm(contract_addrs):
    if addr != ".DS_Store":
        try:
            doc_token_dir = base_dir + addr         
            doc_norm = Statement_Normalizer(doc_token_dir)
            doc_norm.store()
            print("success:",addr)
        except:
            continue
