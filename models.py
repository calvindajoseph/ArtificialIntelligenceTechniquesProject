import os
import time
import copy
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

import global_variables
import dataset_utils
from dataset import TextDataset, BERTDataset, BERTDataLoaderManager

from transformers import logging
logging.set_verbosity_error()

class CVClassifier():
    """
    Train an sklearn-like models with cross validation technique.
    
    Parameters
    ----------
    
    classifier : sklearn - like classifier model.
        An sklearn or similar structured model.
        
    n_folds : int, default = 5
        The number of folds.
    
    smote : bool, default = True
        If true, each training set in a fold will be oversampled using SMOTE
        technique.
    
    Attributes
    ---------
    
    fold_results : pandas.DataFrame
        A pandas dataframe that contains accuracy and weigthed f1 score from
        each fold.
    
    final_results : dict
        A dictionary with the average accuracy and weighted f1 score, as well
        as the standard deviation.
    
    y_trues : numpy.ndarray
        List of all true labels during training. Used for confusion matrix.
    
    y_preds : numpy.ndarray
        List of all predicted labels during training. Used for confusion
        matrix.
    
    """
    
    def __init__(self, classifier, n_folds : int = 5, smote : bool = True):
        self.classifier = classifier
        self.n_folds = n_folds
        if smote:
            self._sampler = SMOTE()
        else:
            self._sampler = None
        self._kfold_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
        self._vectorizer = None
        self.fold_results = None
        self.final_results = None
        self.y_trues = []
        self.y_preds = []
    
    def fit(self, X_data, y_data, tfidf_vectorizer = None):
        """
        Fit a data for cross validation training.
        
        Parameters
        ----------
        
        X_data : list of str
            A list of filtered text. Not the bag of words representation.
        
        y_data : numpy.ndarray or list of integers
            A list of the labels.
        
        tfidf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer,
            default = None
            The vectorizer to be used. If None, then the method will create
            a new vectorizer.
        
        """
        # Create vectorizer.
        if tfidf_vectorizer is None:
            self._vectorizer = TfidfVectorizer()
            self._vectorizer.fit(X_data)
        else:
            self._vectorizer = tfidf_vectorizer
        
        # Reset parameters.
        self.fold_results = None
        self.final_results = None
        self.y_trues = []
        self.y_preds = []
        
        # Make empty lists for scores.
        accuracies = []
        f1_scores = []
        
        # Models to store models.
        models = []
        
        for train_index, test_index in self._kfold_splitter.split(
                X_data, y_data):
            
            # Split into training and testing data.
            X_train = [X_data[i] for i in train_index]
            y_train = [y_data[i] for i in train_index]
            X_test = [X_data[i] for i in test_index]
            y_test = [y_data[i] for i in test_index]
            
            # TF-IDF transform.
            X_train = self._vectorizer.transform(X_train)
            X_test = self._vectorizer.transform(X_test)
            
            # Sample with SMOTE.
            if self._sampler is not None:
                X_train, y_train = self._sampler.fit_resample(
                    X_train, y_train)
            
            # Create model.
            clf = copy.deepcopy(self.classifier)
            clf.fit(X_train, y_train)
            
            # Get prediction.
            y_pred = clf.predict(X_test)
            
            # Put in y_trues and y_preds.
            self.y_trues.extend(y_test)
            self.y_preds.extend(y_pred)
            
            # Process accuracy.
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # Process f1 scores.
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
            
            # Store temporary model.
            models.append(clf)
        
        # Data for dataframe.
        data_fold_results = {
            'accuracy' : accuracies,
            'f1_scores' : f1_scores
        }
        
        # Get the result.
        self.fold_results = pd.DataFrame(data=data_fold_results)
        
        # Final result.
        self.final_results = {
            'mean_acc' : np.mean(accuracies),
            'mean_f1' : np.mean(f1_scores),
            'std_acc' : np.std(accuracies),
            'std_f1' : np.std(f1_scores)
        }
        
        # Store median model.
        self.classifier = models[np.argsort(accuracies)[len(accuracies)//2]]
        
        # To numpy.
        self.y_trues = np.ravel(self.y_trues)
        self.y_preds = np.ravel(self.y_preds)
    
    def classification_report(self, y_trues = None, y_preds = None):
        """
        Create a classification report with precision, recall, and f1 score.
        
        Parameters
        ----------
        
        y_trues : numpy.ndarray, default = None
            If an array is given along with y_preds, the method will return
            the classification report for that values. Otherwise it will use
            the values from training.
        
        y_preds : numpy.ndarray, default = None
            If an array is given along with y_trues, the method will return
            the classification report for that values. Otherwise it will use
            the values from training.
        
        Returns
        -------
        
        classification_report : str
            The classification report from sklearn.
        
        """
        if y_trues is None and y_preds is None:
            return classification_report(self.y_trues, self.y_preds)
        else:
            return classification_report(y_trues, y_preds)
    
    def confusion_matrix(
            self, filename = None, y_trues = None, y_preds = None):
        """
        Create a confusione matrix.
        
        Parameters
        ----------
        
        filename : str, default = None
            If a string is given, the method will store the png of the display
            to the given path.
        
        y_trues : numpy.ndarray, default = None
            If an array is given along with y_preds, the method will return
            the confusion matrix for that values. Otherwise it will use
            the values from training.
        
        y_preds : numpy.ndarray, default = None
            If an array is given along with y_trues, the method will return
            the confusion matrix for that values. Otherwise it will use
            the values from training.
        
        """
        if y_trues is None and y_preds is None:
            ConfusionMatrixDisplay.from_predictions(
                self.y_trues, self.y_preds, 
                display_labels=global_variables.LABELS)
        else:
            ConfusionMatrixDisplay.from_predictions(
                y_trues, y_preds, 
                display_labels=global_variables.LABELS)
        if filename is not None:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def predict(self, text):
        """
        Predict a given text.
        
        Parameters
        ----------
        
        text : str
            The text.
        
        Returns
        -------
        
        prediction : str
            An emotion for the prediction.
        
        """
        text = self._vectorizer.transform([text])
        prediction = self.classifier.predict(text)[0]
        return global_variables.LABELS[prediction]

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
        if not isinstance(model, BERTEmotionClassifier):
            self.model = BERTEmotionClassifier(n_classes)
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

class BERTEmotionModelEnsemble():
    
    def __init__(self, model = None, device = None, n_classes = 6):
        self.epochs = None
        self.set_device(device)
        self.history = defaultdict(list)
        self.results = None
        self.test_acc = None
        self.test_f1 = None
        self.set_model(model, n_classes)
        self.n_classes = n_classes
        self.model_paths = []
    
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
        if not isinstance(model, BERTEmotionClassifier):
            self.model = BERTEmotionClassifier(n_classes)
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
    
    def _filter_data(self, dataset, test_size):
        
        random_percentage = random.randint(84, 88) / 100
        
        df = dataset.df_data
        
        X = df['text'].tolist()
        y = df['label'].to_numpy()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y)
        
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=random_percentage, shuffle=True,
            stratify=y_train)
        
        df_train = pd.DataFrame({'text' : X_train, 'label' : y_train})
        df_val = pd.DataFrame({'text' : X_val, 'label' : y_val})
        
        dataset_train = TextDataset(df_train)
        dataset_val = TextDataset(df_val)
        
        return dataset_train, dataset_val, random_percentage
    
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
    
    def _determine_prediction(self, probabilities):
        return np.argmax(np.array(probabilities))
    
    def _get_predictions(self, probabilities):
        predicted_values = []
        
        for probability in probabilities:
            predicted_value = self._determine_prediction(probability)
            predicted_values.append(predicted_value)
        
        return predicted_values
    
    def _calculate_accuracies(self, probabilities, true_values):
        predicted_values = self._get_predictions(probabilities)
        
        correct = 0
        
        for pred, true, in zip(predicted_values, true_values):
            if pred == true:
                correct += 1
            
        return correct / len(true_values)
    
    def fit(
            self, train_dataset, val_dataset, test_dataset, max_model = 25, 
            max_epochs = 3, tokenizer = None, learning_rate = 5e-5, 
            directory = None, print_console = True):
        
        # Set directory.
        if directory is None:
            directory = 'models/bert_models'
            
        # Save number of epoch.
        if self.epochs is None:
            self.epochs = max_epochs
            
        # If tokenizer none.
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(
                global_variables.PRE_TRAINED_BERT_MODEL_NAME)
            
        # Loss function.
        loss_fn = nn.CrossEntropyLoss().to(self._device)
        
        # Set the optimizer.
        optimizer = AdamW(
            self.model.parameters(), lr = learning_rate, weight_decay=1e-5)
        
        # Create val dataset.
        dataset_val = BERTDataset(
            val_dataset.get_texts(), 
            val_dataset.get_labels(), 
            tokenizer)
        
        # Create test dataset.
        dataset_test = BERTDataset(
            test_dataset.get_texts(), 
            test_dataset.get_labels(), 
            tokenizer)
        
        # Create the dataloader.
        val_data_loader = DataLoader(
            dataset_val,
            batch_size=global_variables.BATCH_SIZE_EVALUATION,
            num_workers=0)
        test_data_loader = DataLoader(
            dataset_test,
            batch_size=global_variables.BATCH_SIZE_EVALUATION,
            num_workers=0)
        
        # Create logs.
        logs = ''
        
        # Create a list to store validation accuracies.
        val_accuracies_all = []
        
        # Create a model paths.
        model_paths = []
        
        # Validation outputs.
        validation_outputs = None
        validation_true_values = None
        
        for iteration in range(max_model):
            
            # Paperwork for iteration.
            start_iteration_str = (
                f'Iteration: {iteration + 1}/{max_model}')
            logs += start_iteration_str + '\n'
            logs += ('-' * 10) + '\n'
            if print_console:
                print(start_iteration_str)
                print('-' * 10)
            
            # Get temporary dataset.
            dataset_train_temp, dataset_val_temp, percentage = self._filter_data(
                train_dataset, 0.1)
            
            # Get lengths.
            n_examples_train = len(dataset_train_temp.df_data.index)
            n_examples_val_temp = len(dataset_val_temp.df_data.index)
            
            # Create train dataset.
            dataset_train_temp = BERTDataset(
                dataset_train_temp.get_texts(), 
                dataset_train_temp.get_labels(), 
                tokenizer)
            
            # Create val dataset.
            dataset_val_temp = BERTDataset(
                dataset_val_temp.get_texts(), 
                dataset_val_temp.get_labels(), 
                tokenizer)
            
            # Create the dataloader.
            train_data_loader = DataLoader(
                dataset_train_temp, 
                batch_size=global_variables.BATCH_SIZE_TRAIN, num_workers=0)
            val_data_loader_temp = DataLoader(
                dataset_val_temp,
                batch_size=global_variables.BATCH_SIZE_EVALUATION,
                num_workers=0)
            
            # Create a list to store validation accuracy.
            val_accuracies = []
            
            # Reset model.
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
            
            # Set a current_path
            current_path = ''
            
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
                    val_data_loader_temp,
                    loss_fn,
                    n_examples_val_temp
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
                
                # Check whether improving.
                val_accuracy = val_acc.to('cpu').numpy()
                if (len(val_accuracies) >= 2):
                    improving = True
                    for i in np.arange(-2, 0, 1, dtype=int):
                        if val_accuracies[i] >= val_accuracy:
                            improving = False
                else:
                    improving = True
                
                # If improving, continue, otherwise break the code.
                if improving:
                    val_accuracies.append(val_accuracy)
                else:
                    # End time
                    end_time_epoch = time.strftime(
                        '%H:%M:%S', time.gmtime(time.time() - start_epoch))
                    end_epoch_str = f'Epoch {epoch + 1} runtime: {end_time_epoch}'
                    logs = logs + end_epoch_str + '\n\n'
                    # Print ending.
                    if print_console:
                        print(end_epoch_str)
                        print()
                    
                    # End training.
                    stop_training = 'Stopped training due to no improvement.'
                    logs += stop_training + '\n\n'
                    if print_console:
                        print(stop_training)
                        print()
                        
                    # Stop epoch.
                    break
                
                # Save the model.
                filename = dataset_utils.get_filename_datetime(
                    directory,
                    f'model_{iteration + 1}_epoch{epoch + 1}_lr{learning_rate}', 'pth')
                current_path = filename
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
            
            # Append last model to paths.
            model_paths.append(current_path)
            
            # Get predictions.
            _, _, prediction_probabilities, real_values = self.get_predictions(
                val_data_loader)
            
            # Load to cpu.
            try:
                prediction_probabilities = prediction_probabilities.detach().numpy()
                real_values = real_values.detach().numpy()
            except:
                try:
                    prediction_probabilities = prediction_probabilities.numpy()
                    real_values = real_values.numpy()
                except:
                    pass
            
            # Update the outputs
            if validation_outputs is None:
                validation_outputs = prediction_probabilities
            else:
                for i, probs in enumerate(prediction_probabilities):
                    for j, prob in enumerate(probs):
                        validation_outputs[i, j] += prob
            
            # Get the true labels.
            if validation_true_values is None:
                validation_true_values = real_values
            
            # Get accuracy.
            val_accuracy = self._calculate_accuracies(
                validation_outputs, validation_true_values)
            
            # Check whether improving.
            if (len(val_accuracies_all) >= 3):
                improving = True
                if (val_accuracies_all[-3] >= val_accuracy) and (
                        val_accuracies_all[-2] >= val_accuracy) and (
                            val_accuracies_all[-1] >= val_accuracy):
                    improving = False
            else:
                improving = True
            
            # Finish iteration.
            end_iteration = f'Iteration validation accuracy: {val_accuracy}'
            logs += end_iteration + '\n\n'
            if print_console:
                print(end_iteration)
                print()
            
            # If improving.
            if improving:
                val_accuracies_all.append(val_accuracy)
            else:
                stop_training = 'Stopped training due to no improvement.'
                logs += stop_training + '\n\n'
                if print_console:
                    print(stop_training)
                    print()
                break
        
        # Save model paths.
        self.model_paths = model_paths
        
        # Save to logs.
        logs += 'Model paths:\n'
        for model_path in self.model_paths:
            logs += model_path + '\n'
        logs += '\n'
        
        # Test accuracy.
        test_predictions, test_true_values = self.predict(test_data_loader)
        self.test_acc = accuracy_score(test_true_values, test_predictions)
        self.test_f1 = f1_score(
            test_true_values, test_predictions, average='weighted')
        
        logs += '\n'
        logs += classification_report(test_true_values, test_predictions)
        logs += '\n'
        
        if print_console:
            print(classification_report(test_true_values, test_predictions))
            print()
        
        # Save logs.
        with open(directory + 'logs.txt', 'w') as f:
            f.write(logs)
    
    def predict(self, data_loader):
        
        probabilities = None
        true_values = None
        
        for model_path in self.model_paths:
            # Load the model.
            self.load_model(model_path, self.n_classes)
            
            # Get predictions.
            _, _, prediction_probabilities, real_values = self.get_predictions(
                data_loader)
            
            # Load to cpu.
            try:
                prediction_probabilities = prediction_probabilities.detach().numpy()
                real_values = real_values.detach().numpy()
            except:
                try:
                    prediction_probabilities = prediction_probabilities.numpy()
                    real_values = real_values.numpy()
                except:
                    pass
            
            # Update the outputs
            if probabilities is None:
                probabilities = prediction_probabilities
            else:
                for i, probs in enumerate(prediction_probabilities):
                    for j, prob in enumerate(probs):
                        probabilities[i, j] += prob
            
            # Get the true labels.
            if true_values is None:
                true_values = real_values
        
        predicted_values = self._get_predictions(probabilities)
        
        return predicted_values, true_values
    
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
    
    def load_paths(self, filename):
        with open(filename, 'r') as f:
            model_paths = f.readlines()
        self.model_paths = model_paths

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

class BERTEmotionModelStackedReloaded():
    """
    A class ONLY used for predicting or using a final model.
    
    Parameters
    ----------
    
    filename : str
        The filename where the model is stored. Use a .pth file.
    """
    
    def __init__(self, filename_main, filename_binary):
        self.set_device()
        self.set_tokenizer()
        self.load_model(filename_main, filename_binary)
    
    def set_device(self):
        """
        Set the device.
        """
        try:
            del self._device
        except:
            pass
        self._device = 'cpu'
    
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
    
    def set_model_main(self):
        """
        Set the model.
        """
        try:
            del self._model_main
        except:
            pass
        self._model_main = BERTEmotionClassifier(
            len(global_variables.LABEL_DESCRIPTION.keys()))
        self._model_main = self._model_main.to(self._device)
    
    def set_model_binary(self):
        """
        Set the model.
        """
        try:
            del self._model_binary
        except:
            pass
        self._model_binary = BERTEmotionClassifier(2)
        self._model_binary = self._model_binary.to(self._device)
    
    def load_model(self, filename_main, filename_binary):
        """
        Load a stored model in pytorch format.
        
        Parameters
        ----------
        
        filename : str
            The directory of a .pth file.
        """
        self.set_model_main()
        self.set_model_binary()
        self._model_main.load_state_dict(
            torch.load(filename_main, map_location=torch.device('cpu')))
        self._model_binary.load_state_dict(
            torch.load(filename_binary, map_location=torch.device('cpu')))
    
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
        
        # Get the main model.
        with torch.no_grad():
            input_ids = encoded_text['input_ids'].to(self._device)
            attention_mask = encoded_text['attention_mask'].to(self._device)
            
            output = self._model_main(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
        
        # Process the main model output
        predictions = np.ravel(MinMaxScaler().fit_transform(
            np.ravel(output.to('cpu').numpy()).reshape(-1, 1)))
        prediction_sum = np.sum(predictions)
        coefficients = {}
        
        # Get labels.
        labels = global_variables.LABEL_DESCRIPTION
        
        # Coefficients.
        for i, label in enumerate(labels.values()):
            coefficients[label] = predictions[i] / prediction_sum
        
        with torch.no_grad():
            input_ids = encoded_text['input_ids'].to(self._device)
            attention_mask = encoded_text['attention_mask'].to(self._device)
            
            output = self._model_binary(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
        
        predictions = np.ravel(MinMaxScaler().fit_transform(
            np.ravel(output.to('cpu').numpy()).reshape(-1, 1)))
        prediction_sum = np.sum(predictions)
        binary_output = predictions[0]
        
        confidence_values = np.fromiter(coefficients.values(), dtype=float)
        X = confidence_values
        X = np.append(X, [binary_output])
        
        logistic_model = joblib.load(global_variables.DIR_FINAL_LOGISTIC_STACK)
        
        final_output = logistic_model.predict(X.reshape(1, -1))
        
        result = {
            'text' : text,
            'emotion' : labels[final_output[0]],
            'coefficients' : coefficients
        }
        
        return result