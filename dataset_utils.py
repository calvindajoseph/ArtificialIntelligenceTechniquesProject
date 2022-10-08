import pandas as pd
from datetime import datetime

import global_variables
from dataset import TextDataset

def import_resampled():
    return TextDataset(pd.read_csv(global_variables.DIR_RESAMPLED))

def import_archive(include = 'all'):
    if not isinstance(include, list):
        if isinstance(include, str):
            if not include == 'all':
                include = 'all'
    
    if include == 'all':
        dataset_train = TextDataset(
            pd.read_csv(global_variables.DIR_ARCHIVE_TRAIN))
        dataset_test = TextDataset(
            pd.read_csv(global_variables.DIR_ARCHIVE_TEST))
        dataset_validation = TextDataset(
            pd.read_csv(global_variables.DIR_ARCHIVE_VALIDATION))

        dataset_train.append_dataset(dataset_test)
        dataset_train.append_dataset(dataset_validation)

        dataset = dataset_train
        
        return dataset
    
    include_dict = {
        'train' : global_variables.DIR_ARCHIVE_TRAIN,
        'test' : global_variables.DIR_ARCHIVE_TEST,
        'validation' : global_variables.DIR_ARCHIVE_VALIDATION
    }
    
    dataset = None
    
    for include_data in include_dict:
        if include_data in include:
            if dataset is None:
                dataset = TextDataset(pd.read_csv(include_dict[include_data]))
            else:
                new_dataset = TextDataset(pd.read_csv(include_dict[include_data]))
                dataset.append_dataset(new_dataset)
            
    return dataset

def get_sample_text(tokenizer):
    sample_text = 'when was i last outside i am stuck at home for 2 weeks'
    encoding = tokenizer.encode_plus(
        sample_text,
        max_length=global_variables.MAX_LENGTH,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding

def get_filename_datetime(directory, name, extension):
    filename = directory + (
        f'/{datetime.now().day}_{datetime.now().month}_{datetime.now().year}')
    filename = filename + '_' + str(name) + '.' + str(extension)
    return filename

def get_datetime():
    return (
        f'{datetime.now().day}_{datetime.now().month}_{datetime.now().year}')