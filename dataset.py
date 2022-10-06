import re
import typing
import numpy as np
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import global_variables

class Dataset:
    
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
    
    def _validate_string(self, string, list_values, default = None):
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
    
    def get_word_lengths(self, source : str = 'raw'):
        source = self._validate_string(source, ['raw', 'texts'], 'raw')
        if self._filled and source == 'raw':
            lengths = np.empty(len(self.df_data.index), dtype=int)
            for i, text in enumerate(self.df_data['text'].tolist()):
                lengths[i] = len(text.strip().split())
            return lengths
        elif self.texts is not None and source == 'texts':
            lengths = np.empty(len(self.texts), dtype=int)
            for i, text in enumerate(self.texts):
                lengths[i] = len(text)
            return lengths
        else:
            return None
    
    def _preprocess_text(self, text):
        text = text.lower()
        text = text.strip()
        text = re.sub('[^a-zA-Z ]', '', text)
        text = re.sub('im', 'i am', text)
        text = re.sub('i m', 'i am', text)
        text = re.sub('ive', 'i have', text)
        text = re.sub('didnt', 'did not', text)
        text = re.sub('it s', 'it is', text)
        text = re.sub('http', '', text)
        words = text.split()
        return words
    
    def _preprocess_texts(self):
        if self._filled:
            max_len = np.max(self.get_word_lengths())
            
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
        if self._n_gram == 'unigram':
            return None
        phrases = Phrases(self.texts, min_count=5, threshold=100)
        phrases_mod = Phraser(phrases)
        if self._n_gram == 'bigram':
            self.texts = [phrases_mod[doc] for doc in self.texts]
            return None
        phrases = Phrases(phrases[self.texts], threshold=100)
        phrases_mod = Phraser(phrases)
        if self._n_gram == 'trigram':
            self.texts = [phrases_mod[doc] for doc in self.texts]
            return None
    
    def get_texts(self, source : str = 'raw'):
        source = self._validate_string(source, ['raw', 'texts'], 'raw')
        if self._filled and source == 'raw':
            return self.df_data['text'].tolist()
        elif self.texts is not None and source == 'texts':
            return self.texts
        else:
            return None
    
    def get_labels(self, label_type : str = 'int'):
        if self._filled:
            label_type = self._validate_string(
                label_type, ['int', 'str'], 'int')
            if label_type == 'int':
                return self.df_data['label'].to_numpy()
            elif label_type == 'str':
                return self.df_data['label'].map(
                    global_variables.LABEL_DESCRIPTION).tolist()
    
    def append_dataset(self, dataset):
        if dataset.validate():
            if self._filled:
                self.df_data = pd.concat(
                    [self.df_data, dataset.df_data], ignore_index=True)
                if self.texts is not None and dataset.texts is not None:
                    self.texts = self.texts + dataset.texts
                return self
        return None