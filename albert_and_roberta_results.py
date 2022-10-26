from transformers import AlbertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score

import global_variables
from dataset import BERTDataLoaderManager
from model_extensions import ROBERTAEmotionModel, ALBERTEmotionModel

# Tokenizers.
albert_tokenizer = AlbertTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_ALBERT_MODEL_NAME)
roberta_tokenizer = RobertaTokenizer.from_pretrained(
    global_variables.PRE_TRAINED_ROBERTA_MODEL_NAME)

# Dataloader managers.
albert_dataloader_manager = BERTDataLoaderManager(
    'preset', albert_tokenizer)
roberta_dataloader_manager = BERTDataLoaderManager(
    'preset', roberta_tokenizer)

# Roberta model.
roberta_model = ROBERTAEmotionModel()
roberta_model.load_model(
    'models/bert_models/26_10_2022/roberta/26_10_2022_epoch2_lr5e-05.pth')

# Get y_pred, y_true.
_, roberta_y_pred, _, roberta_y_true = roberta_model.get_predictions(
    roberta_dataloader_manager.test_data_loader)

# Load to cpu.
roberta_y_pred = roberta_y_pred.detach().numpy()
roberta_y_true = roberta_y_true.detach().numpy()

# Find accuracy and f1 score.
roberta_accuracy = accuracy_score(roberta_y_true, roberta_y_pred)
roberta_f1_score = f1_score(
    roberta_y_true, roberta_y_pred, average='weighted')

# Albert model.
roberta_model = ALBERTEmotionModel()
roberta_model.load_model(
    'models/bert_models/26_10_2022/albert/26_10_2022_epoch2_lr5e-05.pth')

# Get y_pred, y_true.
_, albert_y_pred, _, albert_y_true = roberta_model.get_predictions(
    albert_dataloader_manager.test_data_loader)

# Load to cpu.
albert_y_pred = albert_y_pred.detach().numpy()
albert_y_true = albert_y_true.detach().numpy()

# Find accuracy and f1 score.
albert_accuracy = accuracy_score(albert_y_true, albert_y_pred)
albert_f1_score = f1_score(
    albert_y_true, albert_y_pred, average='weighted')