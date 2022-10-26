from transformers import RobertaTokenizer

import dataset_utils
import global_variables
from dataset import BERTDataLoaderManager
from model_extensions import ROBERTAEmotionModel

#%% Set the tokenizer.
tokenizer = RobertaTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_ROBERTA_MODEL_NAME)

#%% Data loader.
data_loader_manager_regular = BERTDataLoaderManager(
    'preset', tokenizer)

#%% Initialize BERT model.
model = ROBERTAEmotionModel(n_classes=data_loader_manager_regular.n_classes)

model.fit(data_loader_manager_regular, 2, [5e-5], directory=(
    'models/bert_models/' + dataset_utils.get_datetime() + '/roberta/'))

results = model.results
results.to_csv('regular_results.csv', index = False)