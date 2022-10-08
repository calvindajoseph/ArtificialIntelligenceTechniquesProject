import re
import torch
import typing
import numpy as np
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import global_variables

class TextDataset:
    """
    The dataset object for storing emotion dataset.
    
    Parameters
    ----------
    
    data : pandas.DataFrame, default = None
        A pandas DataFrame of the dataset. Please import the csv file using 
        pandas and pass the DataFrame to this class.
    
    use_stopwords : bool, default = True
        If True, the attribute text will not include English stopwords
        from nltk module.
    
    lemmatize : bool, default = True
        If True, each word in attribute text will be lemmatized.
    
    n_gram : str, default = 'unigram'
        The n_gram mode for the bag of words representation of the text.
        There are three modes: ['unigram', 'bigram', 'trigram'].
    
    Attributes
    ----------
    
    df_data : pandas.DataFrame
        The DataFrame of the dataset.
    
    texts : list of str
        The bag of words representation of the dataset.
    """
    
    def __init__(
            self, data : typing.Optional[pd.DataFrame] = None,
            use_stopwords : bool = True,
            lemmatize : bool = True, 
            n_gram : str = 'unigram'):
        # A flag if the data is not empty.
        self._filled = False
        # Flag for using stopwords.
        self._use_stopwords = use_stopwords
        # Flag for perform lemmatization.
        self._lemmatize = lemmatize
        # Which n_gram to use.
        self.set_n_grams(n_gram)
        # Set the dataset.
        self.set_data(data)
    
    def __len__(self):
        """
        Get the length of the dataset.
        """
        if self._filled:
            return len(self.df_data.index)
        else:
            return None
    
    def __getitem__(self, item):
        """
        Get a row dictionary.
        """
        if self._filled:
            columns = self.df_data.columns
            row_dict = {}
            for column in columns:
                row_dict[column] = self.df_data.at[item, column]
            return row_dict
        else:
            return None
    
    def _validate_string(self, string, list_values, default = None):
        """
        For string parameters, used to validate if the string is correct.
        """
        string = string.lower()
        
        if len(string) == 1:
            for value in list_values:
                if string[0] == value[0]:
                    string = value
        
        if string in list_values:
            return string
        else:
            if default is not None:
                return default
            else:
                return None
    
    def validate(self):
        """
        Check the data is valid.
        
        Returns
        -------
        
        valid : bool
            True if there is a data in the Dataset object, False otherwise.
        """
        return self._filled
    
    def set_data(self, data : pd.DataFrame):
        """
        Set the dataset.
        
        Parameters
        ----------
        
        data : pandas.DataFrame
            The pandas dataframe to hold the data.
        """
        if isinstance(data, pd.DataFrame):
            self.df_data = data
            self._filled = True
            self._preprocess_texts()
            return None
        elif self._filled == True:
            return None
        else:
            self.df_data = None
            self._filled = False
            return None
    
    def set_n_grams(self, n_gram : str = 'unigram'):
        """
        Set the n_gram.
        
        Parameters
        ----------
        
        n_gram : str, default = 'unigram'
            n_gram mode. Possible modes are ['unigram', 'bigram', 'trigram']
        """
        
        n_gram = self._validate_string(
            n_gram, global_variables.N_GRAM_VALUES, 'unigram')
        self._n_gram = n_gram
    
    def get_word_lengths(
            self, source : str = 'raw', edit_abbreviation : bool = True):
        """
        Get the lengths of words in either raw or bag of words representation.
        
        Parameters
        ----------
        
        source : str, default = 'raw'
            Which texts it returns, either 'raw' or 'texts' for the bag of
            words representation.
        
        edit_abbreviation : bool, default = True
            If source = 'raw' and edit_abbreviation = True, the data will
            edit weird abbreviation in the text. Unused for source = 'texts'.
        
        Returns
        -------
        
        lengths : numpy.ndarray
            An array of lengths.
        """
        source = self._validate_string(source, ['raw', 'texts'], 'raw')
        if self._filled and source == 'raw':
            lengths = np.empty(len(self.df_data.index), dtype=int)
            for i, text in enumerate(self.df_data['text'].tolist()):
                if edit_abbreviation:
                    text = self._edit_abbreviation(text)
                lengths[i] = len(text.strip().split())
            return lengths
        elif self.texts is not None and source == 'texts':
            lengths = np.empty(len(self.texts), dtype=int)
            for i, text in enumerate(self.texts):
                lengths[i] = len(text)
            return lengths
        else:
            return None
        
    def _edit_abbreviation(self, text):
        """
        Edit weird abbreviation.
        """
        text = re.sub('^im ', 'i am ', text)
        text = re.sub(' im ', ' i am ', text)
        text = re.sub('i m ', 'i am ', text)
        text = re.sub('^ive ', 'i have ', text)
        text = re.sub(' ive ', ' i have ', text)
        text = re.sub('^i d ', 'i had ', text)
        text = re.sub(' i d ', ' i had ', text)
        text = re.sub('^u ', 'you ', text)
        text = re.sub(' u ', ' you ', text)
        text = re.sub('didnt', 'did not', text)
        text = re.sub('can t ', 'can not ', text)
        text = re.sub('n t ', ' not ', text)
        text = re.sub('^it s ', 'it is ', text)
        text = re.sub(' it s ', ' it is ', text)
        text = re.sub('theyre', 'they are', text)
        return text
        
    
    def _preprocess_text(self, text):
        """
        Preprocess raw string for the bag of words.
        """
        text = text.lower()
        text = text.strip()
        text = re.sub('[^0-9a-zA-Z ]', '', text)
        text = self._edit_abbreviation(text)
        text = re.sub(' feeling ', ' ', text)
        text = re.sub(' feel ', ' ', text)
        text = re.sub(' like ', ' ', text)
        text = re.sub(' really ', ' ', text)
        text = re.sub(' get ', ' ', text)
        text = re.sub(' time ', ' ', text)
        text = re.sub(' know ', ' ', text)
        text = re.sub(' people ', ' ', text)
        text = re.sub(' make ', ' ', text)
        text = re.sub(' one ', ' ', text)
        text = re.sub(' would ', ' ', text)
        text = re.sub(' little ', ' ', text)
        text = re.sub(' thing ', ' ', text)
        text = re.sub(' think ', ' ', text)
        text = re.sub('http', '', text)
        words = text.split()
        return words
    
    def _preprocess_texts(self):
        """
        Run this when the dataset is filled. This will transform the data into
        a bag of words representation with a list of strings.
        """
        if self._filled:
            
            if self._use_stopwords:
                stop_words = set(stopwords.words('english'))
            else:
                stop_words = set()
            
            self.texts = []
            for text in self.df_data['text'].tolist():
                words = self._preprocess_text(text)
                words = [w for w in words if not w.lower() in stop_words]
                if self._lemmatize:
                    for i, word in enumerate(words):
                        words[i] = WordNetLemmatizer().lemmatize(word)
                
                self.texts.append(words)
            
            self._process_n_grams()
            
            self.texts
        else:
            self.texts = None
    
    def _process_n_grams(self):
        """
        For the bag of words representation n grams.
        """
        if self._n_gram == 'unigram':
            return None
        phrases = Phrases(self.texts, min_count=5, threshold=5)
        phrases_mod = Phraser(phrases)
        if self._n_gram == 'bigram':
            self.texts = [phrases_mod[doc] for doc in self.texts]
            return None
        phrases = Phrases(phrases[self.texts], threshold=5)
        phrases_mod = Phraser(phrases)
        if self._n_gram == 'trigram':
            self.texts = [phrases_mod[doc] for doc in self.texts]
            return None
    
    def get_texts(
            self,
            source : str = 'raw',
            edit_abbreviation : bool = True,
            bow : bool = True):
        """
        Get either the raw text or the bag of words representation.
        
        Parameters
        ----------
        
        source : str, default = 'raw'
            Which texts it returns, either 'raw' or 'texts' for the bag of
            words representation.
        
        edit_abbreviation : bool, default = True
            If source = 'raw' and edit_abbreviation = True, the data will
            edit weird abbreviation in the text. Unused for source = 'texts'.
        
        Returns
        -------
        
        texts : list of str
            The texts in list.
        """
        source = self._validate_string(source, ['raw', 'texts'], 'raw')
        if self._filled and source == 'raw':
            if edit_abbreviation:
                texts = self.df_data['text'].tolist()
                for i, text in enumerate(texts):
                    texts[i] = self._edit_abbreviation(text)
                return texts
            else:
                return self.df_data['text'].tolist()
        elif self.texts is not None and source == 'texts':
            if bow:
                return self.texts
            else:
                texts = []
                for text in self.texts:
                    combined = ''
                    for word in text:
                        combined += (word + ' ')
                    texts.append(combined.strip())
                return texts
        else:
            return None
    
    def get_labels(self, label_type : str = 'int'):
        """
        Get the labels, either the numerical or the string representation.
        
        Parameters
        ----------
        
        label_type : str, default = 'int'
            Either 'int' or 'str', which dictates what kind of label the
            method will return.
        
        Returns
        -------
        
        labels : numpy.ndarray or list of str
            The labels.
        """
        if self._filled:
            label_type = self._validate_string(
                label_type, ['int', 'str'], 'int')
            if label_type == 'int':
                return self.df_data['label'].to_numpy()
            elif label_type == 'str':
                return self.df_data['label'].map(
                    global_variables.LABEL_DESCRIPTION).tolist()
    
    def append_dataset(self, dataset):
        """
        Append a second Dataset object to this object.
        
        Parameters
        ----------
        
        dataset : Dataset
            The second dataset to append to this object.
        
        Returns
        -------
        
        self : Dataset
            This object.
        """
        if dataset.validate():
            if self._filled:
                self.df_data = pd.concat(
                    [self.df_data, dataset.df_data], ignore_index=True)
                if self.texts is not None and dataset.texts is not None:
                    self.texts = self.texts + dataset.texts
                return self
        return None

class BERTDataset(Dataset):
    """
    A child class of torch.utils.data.Dataset.
    """
    
    def __init__(
            self, texts, labels, tokenizer,
            max_length = global_variables.MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text' : text,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : torch.tensor(label, dtype=torch.long)
        }

class BERTDataLoaderManager():
    """
    Managing torch DataLoader.
    
    Parameters
    ----------
    
    dataset : dataset.TextDataset
        The text dataset, or just the training dataset.
    
    test_size : float, default = 0.1
        The test size.
    
    val_size : float, default = 0.1
        The validation size.
    
    dataset_test : dataset.TextDataset, default = None
        If together wit dataset_val is not None, then it will be the test data.
    
    dataset_val : dataset.TextDataset, default = None
        If together wit dataset_test is not None, then it will be the
        validation data.
    
    Attributes
    ----------
    
    X_train : list of str
        The train data.
    
    X_test : list of str
        The test data.
    
    X_val : list of str
        The validation data.
    
    y_train : numpy.ndarray
        The train labels.
    
    y_test : numpy.ndarray
        The test labels.
    
    y_val : numpy.ndarray
        The validation labels.
    
    train_data_loader : torch.utils.data.DataLoader
        The train data loader.
    
    test_data_loader : torch.utils.data.DataLoader
        The test data loader.
    
    val_data_loader : torch.utils.data.DataLoader
        The validation data loader.
    
    """
    
    def __init__(
            self, dataset : TextDataset, tokenizer, test_size=0.1, 
            val_size=0.1,
            dataset_test : TextDataset = None, 
            dataset_val : TextDataset = None):
        
        if dataset_test is None and dataset_val is None:
            texts = dataset.get_texts()
            labels = dataset.get_labels()
            
            split_one = test_size + val_size
            split_two = val_size / split_one
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                texts, labels, test_size=split_one, stratify=labels, 
                random_state=global_variables.RANDOM_SEED)
            self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
                self.X_test, self.y_test, test_size=split_two, stratify=self.y_test, 
                random_state=global_variables.RANDOM_SEED)
        else:
            self.X_train = dataset.get_texts()
            self.y_train = dataset.get_labels()
            self.X_test = dataset_test.get_texts()
            self.y_test = dataset_test.get_labels()
            self.X_val = dataset_val.get_texts()
            self.y_val = dataset_val.get_labels()
        
        self.train_data_loader = self._create_data_loader(
            self.X_train, self.y_train, tokenizer, 
            global_variables.MAX_LENGTH, 
            global_variables.BATCH_SIZE_TRAIN)
        
        self.test_data_loader = self._create_data_loader(
            self.X_test, self.y_test, tokenizer, 
            global_variables.MAX_LENGTH, 
            global_variables.BATCH_SIZE_EVALUATION)

        self.val_data_loader = self._create_data_loader(
            self.X_val, self.y_val, tokenizer, 
            global_variables.MAX_LENGTH, 
            global_variables.BATCH_SIZE_EVALUATION)
            
        
    def _create_data_loader(
            self, texts, labels, tokenizer, max_len, batch_size):
        """
        Create the data loader.
        """
        dataset = BERTDataset(np.array(texts), labels, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
