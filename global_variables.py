# Directories.
DIR_ARCHIVE_TRAIN = 'data/archive/training.csv'
DIR_ARCHIVE_TEST = 'data/archive/test.csv'
DIR_ARCHIVE_VALIDATION = 'data/archive/validation.csv'

DIR_SPLIT_TRAIN = 'data/split/train.csv'
DIR_SPLIT_TEST = 'data/split/test.csv'
DIR_SPLIT_VALIDATION = 'data/split/val.csv'

DIR_SPLIT_BINARY1_TRAIN = 'data/split/train_binary_one.csv'
DIR_SPLIT_BINARY1_TEST = 'data/split/test_binary_one.csv'
DIR_SPLIT_BINARY1_VALIDATION = 'data/split/val_binary_one.csv'

DIR_SPLIT_BINARY2_TRAIN = 'data/split/train_binary_two.csv'
DIR_SPLIT_BINARY2_TEST = 'data/split/test_binary_two.csv'
DIR_SPLIT_BINARY2_VALIDATION = 'data/split/val_binary_two.csv'

DIR_RESAMPLED = 'data/resampled.csv'
DIR_FINAL_MODEL_BERT = 'models/bert_models/final_model/final_model.pth'
DIR_FINAL_MODEL_BERT_BINARY = 'models/bert_models/final_model/final_model_binary.pth'
DIR_FINAL_LOGISTIC_STACK = 'models/bert_models/final_model/stack_output.joblib'

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

# Roberta.
PRE_TRAINED_ROBERTA_MODEL_NAME = 'roberta-base'

# Albert.
PRE_TRAINED_ALBERT_MODEL_NAME = 'albert-base-v2'