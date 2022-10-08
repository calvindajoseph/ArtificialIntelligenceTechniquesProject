import os
import re
import pickle
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import dataset_utils
from models import CVClassifier

# Create directory.
try:
    os.mkdir('figures/cm')
except:
    pass
try:
    os.mkdir('figures/cm/' + dataset_utils.get_datetime())
except:
    pass

dict_data = defaultdict(list)
dict_fold_results = defaultdict(list)
data = dataset_utils.import_archive()

X = data.get_texts(source='texts', bow=False)
y = data.get_labels()
vectoorizer = TfidfVectorizer()
vectoorizer.fit(X)

models = {
    'dt' : DecisionTreeClassifier(),
    'lr' : LogisticRegression(),
    'nb' : MultinomialNB(),
    'svm' : SVC(),
    'rf' : RandomForestClassifier(),
    'xgb' : XGBClassifier()
}

def train_model(model, str_model, X, y, vectorizer, dict_data, cm_filename):
    print(f'Training model: {str_model}')
    clf = CVClassifier(model)
    clf.fit(X, y, vectorizer)
    fold_results = clf.fold_results
    for col in fold_results.columns:
        dict_fold_results[f'{str_model}_{col}'] = fold_results[col].to_numpy()
    results = clf.final_results
    print(clf.classification_report())
    clf.confusion_matrix(filename=cm_filename)
    dict_data['model'].append(str_model)
    for key in results.keys():
        dict_data[key].append(results[key])
    pickle.dump(clf, open(
        f'{dataset_utils.get_datetime()}_{str_model}.pickle', 'wb'))
    return dict_data

for model_name in models.keys():
    model = models[model_name]
    str_model = model_name
    filename = 'figures/cm/' + dataset_utils.get_datetime(
        ) + f'{str_model}.png'
    dict_data = train_model(
        model, str_model, X, y, vectoorizer, dict_data, filename)

df_results = pd.DataFrame(dict_data)
df_results.to_csv('classical_model_training.csv')
df_folds = pd.DataFrame(dict_fold_results)
df_folds.to_csv('acc_f1.csv')

acc_colnames = [
    colname if bool(
        re.search('acc', colname)) else None for colname in df_folds.columns]
acc_colnames = list(filter(None, acc_colnames))
f1_colnames = [
    colname if bool(
        re.search('f1', colname)) else None for colname in df_folds.columns]
f1_colnames = list(filter(None, f1_colnames))

df_acc = df_folds.filter(items=acc_colnames, axis=1)
df_f1 = df_folds.filter(items=f1_colnames, axis=1)

df_acc = pd.melt(df_acc)
for i in df_acc.index:
    df_acc.at[i, 'variable'] = re.sub('_accuracy', '', df_acc.at[i, 'variable'])
df_f1 = pd.melt(df_f1)
for i in df_f1.index:
    df_f1.at[i, 'variable'] = re.sub('_f1_scores', '', df_f1.at[i, 'variable'])
sns.boxplot(data=df_acc, x='variable', y='value')
plt.savefig('figures/classical_models_acc.png', dpi=300)
sns.boxplot(data=df_f1, x='variable', y='value')
plt.savefig('figures/classical_models_f1.png', dpi=300)