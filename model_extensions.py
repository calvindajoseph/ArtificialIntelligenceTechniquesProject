import time
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn
from torch.optim import AdamW
from transformers import RobertaModel, AlbertModel, get_linear_schedule_with_warmup

import global_variables
import dataset_utils
from dataset import BERTDataLoaderManager

from transformers import logging
logging.set_verbosity_error()

class ROBERTAClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            global_variables.PRE_TRAINED_ROBERTA_MODEL_NAME)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).values()
        output = self.drop(pooled_output)
        return self.out(output)
    
class ALBERTClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(ALBERTClassifier, self).__init__()
        self.albert = AlbertModel.from_pretrained(
            global_variables.PRE_TRAINED_ALBERT_MODEL_NAME)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.albert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).values()
        output = self.drop(pooled_output)
        return self.out(output)

class ROBERTAEmotionModel():
    def __init__(self, model = None, device = None, n_classes = 6):
        self.epochs = None
        self.set_device(device)
        self.history = defaultdict(list)
        self.results = None
        self.test_acc = None
        self.set_model(model, n_classes)
        self.n_classes = n_classes
    
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
    
    def set_model(self, model, n_classes):
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
        if not isinstance(model, ROBERTAClassifier):
            self.model = ROBERTAClassifier(n_classes)
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
        if isinstance(data_loader_manager, BERTDataLoaderManager):
            train_data_loader = data_loader_manager.train_data_loader
            test_data_loader = data_loader_manager.test_data_loader
            val_data_loader = data_loader_manager.val_data_loader
        else:
            train_data_loader = data_loader_manager[0]
            test_data_loader = data_loader_manager[1]
            val_data_loader = data_loader_manager[2]
            
        # Get the number of examples.
        n_examples_train = len(data_loader_manager.X_train)
        n_examples_test = len(data_loader_manager.X_test)
        n_examples_val = len(data_loader_manager.X_val)
        
        # Loss function.
        loss_fn = nn.CrossEntropyLoss().to(self._device)
        
        # Start logs.
        logs = ''
        
        for learning_rate in learning_rates:
            
            self.set_model(None, self.n_classes)
            
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
                    dataset_utils.create_directory(directory)
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
    
    def load_model(self, filename, n_classes = 6):
        """
        Load a stored model in pytorch format.
        
        Parameters
        ----------
        
        filename : str
            The directory of a .pth file.
        """
        self.set_model(None, n_classes)
        self.model.load_state_dict(torch.load(filename))
    
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

class ALBERTEmotionModel():
    def __init__(self, model = None, device = None, n_classes = 6):
        self.epochs = None
        self.set_device(device)
        self.history = defaultdict(list)
        self.results = None
        self.test_acc = None
        self.set_model(model, n_classes)
        self.n_classes = n_classes
    
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
    
    def set_model(self, model, n_classes):
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
        if not isinstance(model, ALBERTClassifier):
            self.model = ALBERTClassifier(n_classes)
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
        if isinstance(data_loader_manager, BERTDataLoaderManager):
            train_data_loader = data_loader_manager.train_data_loader
            test_data_loader = data_loader_manager.test_data_loader
            val_data_loader = data_loader_manager.val_data_loader
        else:
            train_data_loader = data_loader_manager[0]
            test_data_loader = data_loader_manager[1]
            val_data_loader = data_loader_manager[2]
            
        # Get the number of examples.
        n_examples_train = len(data_loader_manager.X_train)
        n_examples_test = len(data_loader_manager.X_test)
        n_examples_val = len(data_loader_manager.X_val)
        
        # Loss function.
        loss_fn = nn.CrossEntropyLoss().to(self._device)
        
        # Start logs.
        logs = ''
        
        for learning_rate in learning_rates:
            
            self.set_model(None, self.n_classes)
            
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
                    dataset_utils.create_directory(directory)
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
    
    def load_model(self, filename, n_classes = 6):
        """
        Load a stored model in pytorch format.
        
        Parameters
        ----------
        
        filename : str
            The directory of a .pth file.
        """
        self.set_model(None, n_classes)
        self.model.load_state_dict(torch.load(filename))
    
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