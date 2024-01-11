from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import classification_report
from sklearn.metrics  import roc_curve
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import time
import re
from imblearn.over_sampling import SMOTE
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
    

y,x = load_data('/home/wangxite/hitework/contract_data/vulnerabilities.txt')

def parseSent(review):
    '''
    Parse text into sentences
    '''
    sentences = []
    for raw_sentence in review:
        if len(raw_sentence) > 0:
            sentences.append(clean_str(str(raw_sentence)))
    return sentences
# Parse each review in the training set into sentences

sentences = []
for review in tqdm(x):
    sentences += parseSent(review)
num_features = 150  #embedding dimension

def makeFeatureVec(review, model, num_features):
    '''
    Transform a review to a feature vector by averaging feature vectors of words
    appeared in that review and in the volcabulary list created
    '''
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index_to_key ) #index2word is the volcabulary list of the Word2Vec model
    isZeroVec = True
    for word in review:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])
            isZeroVec = False
    if isZeroVec == False:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

# #
def getAvgFeatureVecs(reviews, model, num_features):
    '''
    Transform all reviews to feature vectors using makeFeatureVec()
    '''
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
        counter = counter + 1
    return reviewFeatureVecs

print("Training Word2Vec model ...\n")
w2v =  Word2Vec(sentences, vector_size=150, sg=1, min_count=1)
w2v.save("/home/wangxite/ContractCheck/Models/Word2Vec/w2c_model")
w2v=Word2Vec.load("/home/wangxite/ContractCheck/Models/Word2Vec/w2c_model")
# Get feature vectors for training set
x_cleaned = []
for r in x:
    x_cleaned.append(clean_str(str(r)))
    
dataVector = getAvgFeatureVecs(x_cleaned, w2v, num_features)
print("Data set : %d feature vectors with %d dimensions" %dataVector.shape)


output_file = "./Embedded/ouput/Word2Vec_vector.txt"
with open(output_file, "w") as f:
    for label, vector in zip(y, dataVector):
        line = str(label) + " " + " ".join(map(str, vector))
        f.write(line + "\n")






