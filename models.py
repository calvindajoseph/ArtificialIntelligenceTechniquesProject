import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

import global_variables
import dataset_utils

from transformers import logging
logging.set_verbosity_error()

class BERTEmotionClassifier(nn.Module):
    """
    Child class for the nn.Module, for typical pytorch network.
    """
    
    def __init__(self, n_classes):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            global_variables.PRE_TRAINED_BERT_MODEL_NAME)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).values()
        output = self.drop(pooled_output)
        return self.out(output)

class BERTEmotionModel():
    """
    The main class for BERT model training and evaluation. Please keep in mind
    that most of the attributes could be None if this object has not been
    fitted.
    
    It is possible to not fit this model depending on the purpose. If you have
    the model ready in a .pth file, simply use method load_model(). However,
    by doing so most of the attributes will be None.
    
    Parameters
    ----------
    
    model : torch.nn.Module class or child class, default = None
        The main model. If None, a default will be initialized.
    
    device : torch.device, default = None
        Where to store the model, either CPU or GPU. If None, it will look for
        GPU, and then CPU if it does not find GPU.
        
    Attributes
    ----------
    
    model : torch.nn.Module class or child class
        The main model.
    
    epochs : int
        The number of epochs it has been trained on.
    
    history : dict
        A dictionary that stores training information.
    
    results : pandas.DataFrame
        Similar to dictionary, but in pandas DataFrame.
    
    test_acc = float
        The test accuracy.
    """
    
    def __init__(self, model = None, device = None):
        self.epochs = None
        self.set_device(device)
        self.history = defaultdict(list)
        self.results = None
        self.test_acc = None
        self.set_model(model)
    
    def set_device(self, device):
        """
        Set the device.
        
        Parameters
        ----------
        
        device : torch.device
            The device.
        """
        try:
            del self._device
        except:
            pass
        if not isinstance(device, torch.device):
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
    
    def set_model(self, model):
        """
        Set the model.
        
        Parameters
        ----------
        
        model : torch.nn.Module class or child class
            The main model.
        """
        try:
            del self.model
        except:
            pass
        if not isinstance(model, BERTEmotionClassifier):
            self.model = BERTEmotionClassifier(
                len(global_variables.LABEL_DESCRIPTION.keys()))
        else:
            self.model = model
        self.model = self.model.to(self._device)
    
    def _train_epoch(
            self, data_loader, loss_fn, 
            optimizer, scheduler, n_examples):
        """
        Train one epoch.
        """
        self.model = self.model.train()
        
        losses = []
        correct_predictions = 0
        
        for data in data_loader:
            input_ids = data['input_ids'].to(self._device)
            attention_mask = data['attention_mask'].to(self._device)
            labels = data['labels'].to(self._device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predictions = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(predictions == labels)
            losses.append(loss.item())
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return correct_predictions.double() / n_examples, np.mean(losses)
    
    def _eval_model(self, data_loader, loss_fn, n_examples):
        """
        Evaluate a data loader.
        """
        self.model = self.model.eval()
        
        losses = []
        correct_predictions = 0
        
        with torch.no_grad():
            for data in data_loader:
                input_ids = data['input_ids'].to(self._device)
                attention_mask = data['attention_mask'].to(self._device)
                labels = data['labels'].to(self._device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, predictions = torch.max(outputs, dim=1)
                
                loss = loss_fn(outputs, labels)
                
                correct_predictions += torch.sum(predictions == labels)
                
                losses.append(loss.item())
        
        return correct_predictions.double() / n_examples, np.mean(losses)
    
    def fit(
            self, 
            data_loader_manager,
            num_epochs,
            learning_rates,
            directory = None,
            print_console = True):
        """
        Train the data and store the results.
        
        Parameters
        ----------
        
        data_loader_manager : dataset.BERTDataLoaderManager
            The data loader manager.
        
        num_epochs : int
            The number of epochs to train.
        
        learning_rates : list of float
            The learning rates to try.
        
        directory : str, default = None
            Where to store the results. If None then it will use 
            'models/bert_models'
        
        print_console : bool, default = True
            If true, the progress will be printed to console.
        """
        # Set directory.
        if directory is None:
            directory = 'models/bert_models'
        
        # Start training time.
        start_training = time.time()
        
        # Save number of epoch.
        if self.epochs is None:
            self.epochs = num_epochs
        
        # Save the data loader.
        train_data_loader = data_loader_manager.train_data_loader
        test_data_loader = data_loader_manager.test_data_loader
        val_data_loader = data_loader_manager.val_data_loader
        
        # Get the number of examples.
        n_examples_train = len(data_loader_manager.X_train)
        n_examples_test = len(data_loader_manager.X_test)
        n_examples_val = len(data_loader_manager.X_val)
        
        # Loss function.
        loss_fn = nn.CrossEntropyLoss().to(self._device)
        
        # Start logs.
        logs = ''
        
        for learning_rate in learning_rates:
            
            self.set_model(None)
            
            # Set the optimizer.
            optimizer = AdamW(
                self.model.parameters(), lr = learning_rate, weight_decay=1e-5)
            
            # Set scheduler.
            total_steps = len(train_data_loader) * global_variables.MAX_EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            for epoch in range(self.epochs):
                
                # Start time.
                start_epoch = time.time()
                
                # Print beginning of the epoch.
                start_epoch_str = (
                    f'Epoch: {epoch + 1}/{self.epochs}; Learning rate: {learning_rate}')
                logs = logs + start_epoch_str + '\n'
                logs = logs + ('-' * 10) + '\n'
                if print_console:
                    print(start_epoch_str)
                    print('-' * 10)
                
                # Forward and backprop.
                train_acc, train_loss = self._train_epoch(
                    train_data_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    n_examples_train
                )
                
                # Print the train results.
                train_results_str = (
                    f'Train loss: {train_loss}; Accuracy: {train_acc}')
                logs = logs + train_results_str + '\n'
                if print_console:
                    print(train_results_str)
                
                # Calculate accuracy for validation.
                val_acc, val_loss = self._eval_model(
                    val_data_loader,
                    loss_fn,
                    n_examples_val
                )
                
                # Print the validation results.
                validation_results_str = (
                    f'Val loss: {val_loss}; Accuracy: {val_acc}')
                logs = logs + validation_results_str + '\n'
                if print_console:
                    print(validation_results_str)
                
                # Save the results.
                self.history[f'train_acc_{learning_rate}'].append(
                    train_acc.to('cpu').numpy())
                self.history[f'train_loss_{learning_rate}'].append(train_loss)
                self.history[f'val_acc_{learning_rate}'].append(
                    val_acc.to('cpu').numpy())
                self.history[f'val_loss_{learning_rate}'].append(val_loss)
                
                # Save the model.
                filename = dataset_utils.get_filename_datetime(
                    directory,
                    f'epoch{epoch + 1}_lr{learning_rate}', 'pth')
                try:
                    torch.save(self.model.state_dict(), filename)
                except:
                    os.mkdir(directory)
                    torch.save(self.model.state_dict(), filename)
                
                # End time
                end_time_epoch = time.strftime(
                    '%H:%M:%S', time.gmtime(time.time() - start_epoch))
                end_epoch_str = f'Epoch {epoch + 1} runtime: {end_time_epoch}'
                logs = logs + end_epoch_str + '\n\n'
                # Print ending.
                if print_console:
                    print(end_epoch_str)
                    print()
        
        # Get results.
        self.results = pd.DataFrame(data=self.history)
        
        # Get final accuracy.
        self.test_acc, _ = self._eval_model(
            test_data_loader, loss_fn, n_examples_test)
        
        # Print accuracy.
        test_results_str = f'Test accuracy: {self.test_acc}'
        logs = logs + test_results_str + '\n'
        if print_console:
            print(test_results_str)
        
        # Print training time.
        end_time_training = time.strftime(
            '%H:%M:%S', time.gmtime(time.time() - start_training))
        end_training_str = f'Final runtime: {end_time_training}'
        logs = logs + end_training_str + '\n'
        with open(directory + 'logs.txt', 'w') as f:
            f.write(logs)
        
        if print_console:
            print(end_training_str)
    
    def get_predictions(self, data_loader):
        """
        Get a set of predictions based on a data loader.
        
        Parameters
        ----------
        
        data_loader : torch.utils.data.DataLoader
            A data loader.
        
        Returns
        -------
        
        texts : list of str
            The texts in the data loader.
        
        predictions : list of int
            The predictions.
        
        prediction_probabilities : list
            The final layer coefficients.
        
        real_values : list of int
            True labels.
        """
        self.model = self.model.eval()
        
        texts = []
        predictions = []
        prediction_probabilities = []
        real_values = []
        
        with torch.no_grad():
            for data in data_loader:
                
                text = data['text']
                input_ids = data['input_ids'].to(self._device)
                attention_mask = data['attention_mask'].to(self._device)
                labels = data['labels'].to(self._device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs, dim=1)
                
                texts.extend(text)
                predictions.extend(preds)
                prediction_probabilities.extend(outputs)
                real_values.extend(labels)
        
        predictions = torch.stack(predictions).cpu()
        prediction_probabilities = torch.stack(prediction_probabilities).cpu()
        real_values = torch.stack(real_values).cpu()
        
        return texts, predictions, prediction_probabilities, real_values
    
    def classification_report(self, data_loader, directory = None):
        """
        Make a classification report.
        
        Parameters
        ----------
        
        data_loader : torch.utils.data.DataLoader
            The data to be tested.
        
        directory : str, default = None
            If a directory is given, the method will store the classification
            reports to that directory.
        
        Returns
        -------
        
        classification_report_str : str
            The main classification report.
        
        df_cm : pandas.DataFrame
            Confusion Matrix.
        """
        texts, predictions, predictions_probabilities, real_values = self.get_predictions(
            data_loader)
        
        classification_report_str = classification_report(
            real_values, predictions, 
            target_names=global_variables.LABEL_DESCRIPTION.values())
        
        print(classification_report_str)
        
        cm = confusion_matrix(real_values, predictions)
        
        df_cm = pd.DataFrame(
            cm, 
            index=global_variables.LABEL_DESCRIPTION.values(), 
            columns=global_variables.LABEL_DESCRIPTION.values())
        
        print(df_cm)
        
        if directory is not None:
            try:
                with open(directory + 'classification_report.txt', 'w') as f:
                    f.write(classification_report_str)
                df_cm.to_csv(directory + 'confusion_matrix.csv')
            except:
                os.mkdir(directory)
                with open(directory + 'classification_report.txt', 'w') as f:
                    f.write(classification_report_str)
                df_cm.to_csv(directory + 'confusion_matrix.csv')
        
        return classification_report_str, df_cm
    
    def load_model(self, filename):
        """
        Load a stored model in pytorch format.
        
        Parameters
        ----------
        
        filename : str
            The directory of a .pth file.
        """
        self.set_model(None)
        self.model.load_state_dict(torch.load(filename))

class BERTEmotionModelReloaded():
    """
    A class ONLY used for predicting or using a final model.
    
    Parameters
    ----------
    
    filename : str
        The filename where the model is stored. Use a .pth file.
    """
    
    def __init__(self, filename):
        self.set_device()
        self.set_tokenizer()
        self.load_model(filename)
    
    def set_device(self):
        """
        Set the device.
        """
        try:
            del self._device
        except:
            pass
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    
    def set_tokenizer(self):
        """
        Set the tokenizer.
        """
        try:
            del self._tokenizer
        except:
            pass
        self._tokenizer = BertTokenizer.from_pretrained(
            global_variables.PRE_TRAINED_BERT_MODEL_NAME)
    
    def set_model(self):
        """
        Set the model.
        """
        try:
            del self._model
        except:
            pass
        self._model = BERTEmotionClassifier(
            len(global_variables.LABEL_DESCRIPTION.keys()))
        self._model = self._model.to(self._device)
    
    def load_model(self, filename):
        """
        Load a stored model in pytorch format.
        
        Parameters
        ----------
        
        filename : str
            The directory of a .pth file.
        """
        self.set_model()
        self._model.load_state_dict(
            torch.load(filename, map_location=torch.device('cpu')))
    
    def predict(self, text):
        """
        Predict a text.
        
        Parameters
        ----------
        
        text : str
            The text to predict.
        
        Returns
        -------
        
        result : dict
            A dictionary of the result.
        """
        text = text.lower()
        
        encoded_text = self._tokenizer.encode_plus(
            text,
            max_length=global_variables.MAX_LENGTH,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoded_text['input_ids'].to(self._device)
            attention_mask = encoded_text['attention_mask'].to(self._device)
            
            output = self._model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
            
        predictions = np.ravel(MinMaxScaler().fit_transform(
            np.ravel(output.to('cpu').numpy()).reshape(-1, 1)))
        prediction_sum = np.sum(predictions)
        coefficients = {}
        
        labels = global_variables.LABEL_DESCRIPTION
        
        for i, label in enumerate(labels.values()):
            coefficients[label] = predictions[i] / prediction_sum
        
        result = {
            'text' : text,
            'emotion' : labels[prediction.to('cpu').numpy()[0]],
            'coefficients' : coefficients
        }
        
        return result