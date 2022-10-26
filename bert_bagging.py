import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformers import BertTokenizer

import global_variables
import dataset_utils
from dataset import BERTDataset
from models import BERTEmotionModel

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_BERT_MODEL_NAME)

df_train = pd.read_csv('data/split/train.csv')
df_test = pd.read_csv('data/split/test.csv')
df_val = pd.read_csv('data/split/val.csv')

dataset_test = BERTDataset(
    df_test['text'].tolist(), df_test['label'].to_numpy(), tokenizer)
dataset_val = BERTDataset(
    df_val['text'].tolist(), df_val['label'].to_numpy(), tokenizer)

dataloader_test = DataLoader(
    dataset_test, batch_size=global_variables.BATCH_SIZE_EVALUATION,
    num_workers=0)
dataloader_val = DataLoader(
    dataset_val, batch_size=global_variables.BATCH_SIZE_EVALUATION,
    num_workers=0)

def split_df(df, n_split):
    df_list = []
    df_temp = df
    size = int(len(df.index) / n_split)
    for i in range(n_split - 1):
        X = df_temp['text'].tolist()
        y = df_temp['label'].to_numpy()
        
        X_left, X_split, y_left, y_split = train_test_split(
            X, y, test_size = size, stratify=y)
        
        df_temp = pd.DataFrame({
            'text' : X_left,
            'label' : y_left
        })
        
        df_current = pd.DataFrame({
            'text' : X_split,
            'label' : y_split
        })
        
        df_list.append(df_current)
    
    df_list.append(df_temp)
    
    return df_list

df_list = split_df(df_train, 5)

models = []

for i, df in enumerate(df_list):
    model = BERTEmotionModel()
    
    dataset_train = BERTDataset(
        df['text'].tolist(), df['label'].to_numpy(), tokenizer)
    
    dataloader_train = DataLoader(dataset_train)
    
    data_loaders = [
        dataloader_train,
        dataloader_test,
        dataloader_val
    ]

    model.fit(data_loaders, 2, [5e-5], directory=(
        'models/bert_models/' + dataset_utils.get_datetime() + f'/ensemble/no_{i}/'))

    models.append(model)
