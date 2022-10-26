import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

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

def import_split():
    dataset_train = TextDataset(
        pd.read_csv(global_variables.DIR_SPLIT_TRAIN))
    dataset_test = TextDataset(
        pd.read_csv(global_variables.DIR_SPLIT_TEST))
    dataset_validation = TextDataset(
        pd.read_csv(global_variables.DIR_SPLIT_VALIDATION))
    return dataset_train, dataset_test, dataset_validation

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

def split_data(dataset, split_ratio : list = [0.8, 0.1, 0.1]):
    df = dataset.df_data
    # Calculate the splitting ratio.
    test_size = split_ratio[1]
    val_size = split_ratio[2]
    first_split_test_size = test_size + val_size
    second_split_test_size = val_size / (first_split_test_size)
    # Create initial data and target.
    X = df['text'].to_list()
    y = df['label'].to_numpy()
    # First split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=first_split_test_size,
        random_state=global_variables.RANDOM_SEED, stratify=y)
    # Second split.
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=second_split_test_size,
        random_state=global_variables.RANDOM_SEED, stratify=y_test)
    # Create the dataset.
    train_dataset = TextDataset(pd.DataFrame(data={
        'text' : X_train, 'label' : y_train}))
    test_dataset = TextDataset(pd.DataFrame(data={
        'text' : X_test, 'label' : y_test}))
    val_dataset = TextDataset(pd.DataFrame(data={
        'text' : X_val, 'label' : y_val}))
    return train_dataset, test_dataset, val_dataset

def create_directory(directory):
    if (directory[-1] == '/'):
        directory = directory[:-1]
    folders = directory.split('/')
    current_directory = ''
    for folder in folders:
        current_directory += folder + '/'
        try:
            os.mkdir(current_directory)
        except:
            pass
    return folders