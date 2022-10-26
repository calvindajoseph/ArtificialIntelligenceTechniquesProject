from transformers import AlbertTokenizer

import dataset_utils
import global_variables
from dataset import BERTDataLoaderManager
from model_extensions import ALBERTEmotionModel

#%% Set the tokenizer.
tokenizer = AlbertTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_ALBERT_MODEL_NAME)

#%% Data loader.
data_loader_manager_regular = BERTDataLoaderManager(
    'preset', tokenizer)

#%% Initialize BERT model.
model = ALBERTEmotionModel(n_classes=data_loader_manager_regular.n_classes)

model.fit(data_loader_manager_regular, 10, [5e-5, 3e-5, 2e-5], directory=(
    'models/bert_models/' + dataset_utils.get_datetime() + '/albert/'))

results = model.results
results.to_csv('regular_results.csv', index = False)