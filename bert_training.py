from transformers import BertTokenizer

import dataset_utils
import global_variables
from dataset import BERTDataLoaderManager
from models import BERTEmotionModel

#%% Get dataset.
dataset = dataset_utils.import_archive()

#%% Set the tokenizer.
tokenizer = BertTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_BERT_MODEL_NAME)

#%% Data loader.
data_loader_manager_regular = BERTDataLoaderManager(dataset, tokenizer)

#%% Initialize BERT model.
model = BERTEmotionModel()

model.fit(data_loader_manager_regular, 10, [5e-5, 3e-5, 2e-5], directory=(
    'models/bert_models/' + dataset_utils.get_datetime() + '/batch_size/'))

results = model.results
results.to_csv('regular_results.csv')