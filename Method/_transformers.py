import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix,matthews_corrcoef

import re
from imblearn.over_sampling import SMOTE

class TextDataset(Dataset):
    def __init__(self, encodings, labels, tokenizer):
        self.encodings = encodings
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_class_weights(labels):
    counter = Counter(labels)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

# 加载数据和tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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
    label = [line.strip().split('\t')[0] for line in open(data_path, "r")]
    label_list = ['__label__none', '__label__ARTHM', '__label__DOS', '__label__TimeM',
                  '__label__TX-Origin','__label__UE']
    label = [label_list.index(_label) for _label in label]
    return label, context

# 加载数据并清洗
y, x = load_data('/home/wangxite/hitework/contract_data/new_vulnerabilities.txt')

# 将数据划分为训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(x, y, test_size=0.2)

# 使用tokenizer将文本转换为tokens
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

# 将数据转换为NumPy数组
train_encodings_array = np.hstack((train_encodings["input_ids"], train_encodings["attention_mask"]))
train_labels = np.array(train_labels)
train_encodings_array = np.array(train_encodings_array)

# 定义SMOTE对象并拟合数据
smote = SMOTE(random_state=42)
train_encodings_array, train_labels = smote.fit_resample(train_encodings_array, train_labels)

# 将数据转换为PyTorch Dataset对象
train_encodings = {"input_ids": train_encodings_array[:,:512],
                   "attention_mask": train_encodings_array[:,512:]}
train_dataset = TextDataset(train_encodings, train_labels, tokenizer)
test_dataset = TextDataset(tokenizer(test_texts, truncation=True, padding=True, max_length=512), test_labels, tokenizer)


# 计算类权重
class_weights = compute_class_weights(train_labels)
class_weights = torch.tensor(list(class_weights.values()))

# 加载预训练的RoBERTa模型
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(set(y)))

# 自定义Trainer类来使用自定义损失函数
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 定义DataCollatorWithPadding类
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 设置训练参数
training_args = TrainingArguments(
    output_dir='/home/wangxite/hitework/transformers_output',
    num_train_epochs=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/home/wangxite/hitework/logs',
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000
)

# 初始化CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 评估模型性能
evaluation_results = trainer.evaluate()
print(evaluation_results)

# 使用训练好的模型进行预测
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# 正确的标签名称列表
label_names = ['__label__none', '__label__ARTHM', '__label__DOS', '__label__TimeM',
               '__label__TX-Origin',]

# 计算并输出每一类的准确率、精度和召回率
report = classification_report(test_labels, preds, target_names=label_names)
print(report)

# 计算混淆矩阵
conf_matrix = confusion_matrix(test_labels, preds)

# 计算并输出每一类的FPR
fpr = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
false_positive_sum = fpr.sum(axis=0)
total_negative = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
fpr = fpr / total_negative
fpr_dict = {label_names[i]: fpr[i] for i in range(len(label_names))}
print("False Positive Rate by class:", fpr_dict)

fpr_total =  conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[1][1])
print(fpr_total)

mcc = matthews_corrcoef(test_labels, preds)


print("Total FPR:", fpr_total)
print("Total MCC:", mcc)