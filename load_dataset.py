import pandas as pd
import numpy as np
import torch
from torchtext import data
from torchtext.vocab import Vectors

def text_tokenize(x):
    return x.split()

def label_tokenize(y):
    return y

#文本的预处理方法
TEXT=data.Field(sequential=True,tokenize=text_tokenize,fix_length=40)

#标签的预处理方法
LABEL=data.Field(sequential=False,tokenize=label_tokenize,use_vocab=False)

def get_dataset(csv_data,text_field,label_field):
    # csv_data为padas读取后的DataFrame
    fields=[('sentence1',text_field),('sentence2',text_field),('gold_label',label_field)]
    examples=[]
    for text1,text2,label in zip(csv_data['sentence1'],csv_data['sentence2'],csv_data['gold_label']):
        examples.append(data.Example.fromlist([text1,text2,label],fields))
    return examples,fields

train_data=pd.read_csv('data/SNLI/snli-train.txt',sep='\t')
test_data=pd.read_csv('data/SNLI/snli-test.txt',sep='\t')

train_examples,train_fields=get_dataset(train_data,TEXT,LABEL)
test_examples,test_fields=get_dataset(test_data,TEXT,LABEL)
train_examples = train_examples[:10000]
test_examples = test_examples[:3000]
print(len(train_examples))
print(len(test_examples))
train=data.Dataset(train_examples, train_fields)
test=data.Dataset(test_examples, test_fields)

vectors = Vectors(name='data/glove.6B.300d.txt')
TEXT.build_vocab(train,vectors=vectors)
#print(len(TEXT.vocab))
#构建迭代器
from torchtext.data import Iterator
train_iter=Iterator(train,batch_size=100,sort=False,device=torch.device('cuda'),repeat=False)
test_iter = Iterator(test, batch_size=100,device=torch.device('cuda'), sort=False,repeat=False)

