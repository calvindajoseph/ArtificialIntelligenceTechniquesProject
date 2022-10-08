import pandas as pd
from collections import Counter

import global_variables
from dataset import TextDataset

def get_top_ngrams(df, n_gram, label, n=20):
    df = df[df['label'] == label].reset_index(drop=True)
    texts = [
        word for text in TextDataset(
            data=df, n_gram=n_gram).get_texts('texts') for word in text]
    counter = Counter(texts)
    top_words = counter.most_common()[:n]
    words = []
    for word in top_words:
        words.append(word[0])
    return words

#%% Import dataset.
dataset_train = TextDataset(pd.read_csv('data/archive/training.csv'))
dataset_test = TextDataset(pd.read_csv('data/archive/test.csv'))
dataset_validation = TextDataset(pd.read_csv('data/archive/validation.csv'))

dataset_train.append_dataset(dataset_test)
dataset_train.append_dataset(dataset_validation)

dataset = dataset_train

del dataset_train
del dataset_test
del dataset_validation

df_data = dataset.df_data

#%% Main loop.

word_dict = {}

for label_index, label in enumerate(global_variables.LABEL_DESCRIPTION.keys()):
    word_list = get_top_ngrams(df_data, 'bigram', label)
    word_dict[f'{global_variables.LABEL_DESCRIPTION[label]}'] = word_list

word_count = {}
freq_word = []
for label in word_dict.keys():
    for word in word_dict[label]:
        if word in word_count:
            word_count[word] += 1
            if word_count[word] >= 5 and word not in freq_word:
                freq_word.append(word)
        else:
            word_count[word] = 1