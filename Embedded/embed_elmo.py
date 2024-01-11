from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
from tqdm import tqdm
import re
import torch
import torch.nn.utils.rnn as rnn_utils
from sklearn.decomposition import PCA

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(data_path):
    # Load data from files
    context = [line.strip().split('\t')[1] for line in open(data_path, "r")]
    #context = [clean_str(str(sent)) for sent in context]
    label = [line.strip().split('\t')[0] for line in open(data_path, "r")]
    label_list = ['__label__none', '__label__ARTHM', '__label__DOS', '__label__TimeM',
                  '__label__TX-Origin']
    label = [label_list.index(_label) for _label in label]
    return label,context
    

# 加载数据并清洗
y, x = load_data('./Embedded/input/vulnerabilities.txt')
x_cleaned = [clean_str(str(sent)).split() for sent in x]

options_file = "./Models/ELMo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "./Models/ELMo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
elmo = elmo.cuda()  


def elmo_embedding_batch(batch):
    character_ids = batch_to_ids(batch)
    character_ids = character_ids.cuda()  
    embeddings = elmo(character_ids)
    return embeddings['elmo_representations'][0].detach().cpu().numpy()

max_length = 150
padded_x = [sent[:max_length] + [''] * (max_length - len(sent)) if len(sent) < max_length else sent[:max_length] for sent in x_cleaned]


batch_size = 8
data_embeddings = []

for i in tqdm(range(0, len(padded_x), batch_size)):
    batch = padded_x[i:i+batch_size]
    batch_embeddings = elmo_embedding_batch(batch)
    data_embeddings.append(batch_embeddings)


data_embeddings = np.vstack(data_embeddings)


num_sentences = len(padded_x)
print(f'Number of sentences: {num_sentences}')
print(f'Shape of data_embeddings: {data_embeddings.shape}')


data_embeddings_reshaped = data_embeddings.reshape(num_sentences, max_length, 256)


pca = PCA(n_components=16)
data_embeddings_pca = pca.fit_transform(data_embeddings_reshaped.reshape(-1, 256))
data_embeddings_64d = data_embeddings_pca.reshape(num_sentences, max_length, 16)


output_file = './Embedded/ouput/elmo_vulnerabilities_vector.txt'
with open(output_file, 'w') as f:
    for i, embedding in enumerate(data_embeddings_64d):
        
        embedding_str = ' '.join(map(str, embedding.flatten()))
   
        f.write(f'{y[i]} {embedding_str}\n')

print(f"ELMo embeddings saved to {output_file}")