import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pickle
from dhg import Hypergraph
import torch
import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph, Hypergraph
from dhg.data import Cora
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

class HypergraphDataset:
    def __init__(self, inputs, outputs,train, k=5):
        self.inputs = inputs.clone().detach().to(dtype=torch.float32) # 转换为 float32
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
        self.k = k
        self.hypergraph, self.graph= self.create_hypergraph(train)


    def create_hypergraph(self,train):

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.inputs)
        distances, indices = nbrs.kneighbors(self.inputs)

        edge_list = []
        for i in range(len(self.inputs)):
            hyperedge = [i] + indices[i, 1:].tolist()  # 忽略自身
            edge_list.append(hyperedge)

        G = Graph(len(self.inputs), edge_list)
        HG = Hypergraph.from_graph_kHop(G, k=1)
        return HG,G

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class Hypergraph1Dataset:
    def __init__(self, text_features, genders, ages,inputs, outputs,train, k=5):
        self.text_features = text_features
        self.genders = genders
        self.ages = ages
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
        self.k = k
        self.hypergraph,self.graph = self.create_hypergraph(train)
        self.inputs = inputs.clone().detach().to(dtype=torch.float32) # 转换为 float32

    def create_hypergraph(self,train):
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.ages)
        distances, indices = nbrs.kneighbors(self.ages)

        edge_list = []
        for i in range(len(self.ages)):
            
            hyperedge = [i] + indices[i, 1:].tolist()  # 忽略自身
            edge_list.append(hyperedge)
        # 创建超图
        G = Graph(len(self.text_features), edge_list)
        HG = Hypergraph.from_graph_kHop(G, k=1)
        return HG,G


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
            # 返回索引idx对应的样本
        return self.inputs[idx], self.outputs[idx]
def load_and_preprocess_data_with_graph(file_path,train, k=11):
    # 读取数据
    df = pd.read_csv(file_path)

    # 假设BERT嵌入保存在'text_feature'列中，并且是字符串形式的数组
    df['text_feature'] = df['text_feature'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

    # 分离特征和目标
    text_features = df['text_feature'].tolist()
    # 文本特征归一化
    text_feature_scaler = MinMaxScaler()
    text_features = text_feature_scaler.fit_transform(text_features)
    # # # 应用PCA降维到60维
    pca = PCA(n_components=66)
    text_features = pca.fit_transform(text_features)

    # 定义所有可能的类别
    gender_categories = ['total-user', '男', '女']
    age_categories = ['total-user', '46-50', '41-45', '36-40', '31-35', '26-30', '21-25', 'under20']
    gender_encoder = OneHotEncoder(categories=[gender_categories])
    gender_onehot = gender_encoder.fit_transform(df[['gender']])
    genders = gender_onehot.toarray()
    # 年龄处理：独热编码
    age_encoder = OneHotEncoder(categories=[age_categories])
    age_onehot = age_encoder.fit_transform(df[['age']])
    ages = age_onehot.toarray()
    outputs = df['answer'].values
    output_scaler = MinMaxScaler()
    outputs = output_scaler.fit_transform(outputs)
    text_features = torch.from_numpy(text_features)
    genders = torch.from_numpy(genders)
    ages = torch.from_numpy(ages)

    inputs = torch.cat((text_features, genders, ages), 1)
    #dataset = HypergraphDataset(inputs, outputs,train, k=k)

    dataset = Hypergraph1Dataset(text_features,genders,ages ,inputs, outputs,train, k=k)
    return dataset