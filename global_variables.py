# Directories.
DIR_ARCHIVE_TRAIN = 'data/archive/training.csv'
DIR_ARCHIVE_TEST = 'data/archive/test.csv'
DIR_ARCHIVE_VALIDATION = 'data/archive/validation.csv'
DIR_RESAMPLED = 'data/resampled.csv'
DIR_FINAL_MODEL_BERT = 'models/bert_models/final_model/final_model.pth'

# Global Variables.
RANDOM_SEED = 0

# Dataset Parameters.
LABEL_DESCRIPTION = {
    0 : 'sadness',
    1 : 'joy',
    2 : 'love',
    3 : 'anger',
    4 : 'fear',
    5 : 'surprise'
}
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

N_GRAM_VALUES = ['unigram', 'bigram', 'trigram']

# BERT Model Parameters
PRE_TRAINED_BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 70
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVALUATION = 16
MAX_EPOCHS = 1