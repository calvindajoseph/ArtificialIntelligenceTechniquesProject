import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import global_variables
from dataset import TextDataset

import warnings
warnings.filterwarnings("ignore")

# Plot parameters.
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
texts = dataset.texts
labels = dataset.get_labels('str')

#%% Label count plot.

label_count_plot = sns.countplot(
    labels, order=pd.Series(labels).value_counts().index)
plt.title('Count Plot of Emotions', fontsize=20)
plt.xlabel('Emotion', fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel('Count', fontsize=16)
plt.yticks(fontsize=14)
plt.savefig('figures/count_plot_emotions.png')
plt.show()

#%% Text lengths.

raw_text_lengths = dataset.get_word_lengths('raw')
text_length_dist_plot = sns.distplot(raw_text_lengths)
plt.title('Text Length Distribution', fontsize=20)
plt.xlabel('Text Length', fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel('Density', fontsize=16)
plt.yticks(fontsize=14)
plt.savefig('figures/text_length_distribution.png')
plt.show()

#%% Text lengths over emotions.
df_emotion_length = pd.DataFrame(data={
    'emotion' : labels, 'length' : raw_text_lengths})
emotion_text_length_boxplot = sns.boxplot(
    data=df_emotion_length, x='emotion', y='length')
plt.title('Text Length Distribution over Emotions', fontsize=20)
plt.xlabel('Emotion', fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel('Text Length', fontsize=16)
plt.yticks(fontsize=14)
plt.savefig('figures/text_length_distribution_over_emotions.png')
plt.show()

#%% N grams.

def get_top_ngrams(df, n_gram, label, n=5):
    df = df[df['label'] == label].reset_index(drop=True)
    texts = [
        word for text in TextDataset(
            data=df, n_gram=n_gram).get_texts('texts') for word in text]
    counter = Counter(texts)
    top_words = counter.most_common()[:n]
    data = {'word' : [], 'count' : []}
    for word in top_words:
        data['word'].append(word[0])
        data['count'].append(word[1])
    df_top_word = pd.DataFrame(data=data)
    return df_top_word

for label_index, label in enumerate(global_variables.LABEL_DESCRIPTION.keys()):
    fig, axes = plt.subplots(3)
    plt.tight_layout()
    for n_gram_index, n_gram in enumerate(global_variables.N_GRAM_VALUES):
        df_top_ngrams = get_top_ngrams(df_data, n_gram, label)
        sns.barplot(
            data=df_top_ngrams, x='word', y='count', 
            ax=axes[n_gram_index])
        axes[n_gram_index].set_title(
            f'{global_variables.LABEL_DESCRIPTION[label]}_{n_gram}')
        axes[n_gram_index].set_xlabel('')
        axes[n_gram_index].set_ylabel('')
    plt.savefig(
        f'figures/{global_variables.LABEL_DESCRIPTION[label]}_topwords.png')
    plt.show()