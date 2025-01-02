import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import load

######## Libraries ################

class DualPerceptron:

    def __init__(self, kernel, epochs=1000, patience=10, time=None, **kwargs):
        self.kernel = kernel
        self.epochs = epochs
        self.patience = patience
        self.alpha = None
        self.b = 0
        self.gram_matrix = None
        self.accuracy = None
        self.time = time

        # Kernel-specific hyperparameters validation
        if self.kernel == 'linear':
            self._validate_hyperparameters(kwargs, [])
        elif self.kernel == 'rbf':
            self.gamma = kwargs.get('gamma', 0.5)
            self._validate_hyperparameters(kwargs, ['gamma'])
        elif self.kernel == 'poly':
            self.d = kwargs.get('d', 2)
            self.c = kwargs.get('c', 1)
            self._validate_hyperparameters(kwargs, ['d', 'c'])
        else:
            raise ValueError('Unsupported kernel type')

    def _validate_hyperparameters(self, kwargs, allowed):
        invalid_keys = set(kwargs.keys()) - set(allowed)
        if invalid_keys:
            raise ValueError(f'Invalid hyperparameters: {invalid_keys}')

    def fit(self, X_train, y_train, X_val, y_val, **params):
        r2 = np.round(np.max(np.linalg.norm(X_train, axis=1))**2, 1)
        self.X_train, self.y_train = X_train, y_train
        self.n = len(X_train)
        self.alpha = np.zeros(self.n)
        self.gram_matrix = self._compute_gram_matrix(X_train)
        train_accuracies = []
        validation_accuracies = []
        best_validation_accuracy = -float('inf')
        best_accuracy = -float('inf')
        patience_counter = 0
        previous_accuracy = 0
        start_time = time.time()

        for epoch in tqdm(range(self.epochs), desc=f'Training model with {self.kernel} kernel'):
            errors = 0
            for i in range(self.n):
                w = sum(self.alpha[j] * y_train[j] * self.gram_matrix[i, j] for j in range(self.n)) + self.b
                if w * y_train[i] <= 0:
                    self.alpha[i] += 1
                    self.b += y_train[i]*r2
                    errors += 1
            epoch_accuracy = 1-(errors / self.n) # Accuratezza nel train

            # Check for early stopping using patience
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                patience_counter = 0  # Reset patience if accuracy improves
            else:
                patience_counter += 1

            previous_accuracy = epoch_accuracy

            # Prediction on validation set
            self.predict(X_val, y_val)
            validation_accuracy = self.get_accuracy()
            validation_accuracies.append(validation_accuracy)
            train_accuracies.append(epoch_accuracy)

            if validation_accuracy > best_validation_accuracy:
                best_model_params = self.get_params()

            # Stop if the accuracy doesn't improve for `patience` consecutive epochs
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in accuracy.")
                break
        
        self.save_params(best_model_params) # Salvo i parametri del modello che hanno datto l'accuracy più alta alla fine delle epoche
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Training time: {elapsed_time} seconds')
        return {'train_accuracies': train_accuracies, 
                'validation_accuracies': validation_accuracies,
                'train_accuracy': max(train_accuracies),
                'validation_accuracy': max(validation_accuracies),
                '_model_params': best_model_params
        }

    def predict(self, X_test, y_test):
        self.predictions = np.array([self._predict_point(x) for x in X_test])
        self.accuracy = np.mean(self.predictions == y_test)
        return self.predictions

    def _predict_point(self, x):
        w = sum(self.alpha[i] * self.y_train[i] * self._kernel(self.X_train[i], x) for i in range(self.n)) + self.b
        return np.sign(w)

    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + self.c) ** self.d
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    def _compute_gram_matrix(self, X):
        n = X.shape[0]
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gram[i, j] = self._kernel(X[i], X[j])
        return gram 
        
    def get_accuracy(self):
        return self.accuracy

    def get_params(self):
        params = {'alpha': self.alpha, 'bias': self.b, 'kernel': self.kernel, 'X_train': self.X_train, 'y_train': self.y_train}
        if self.kernel == 'rbf':
            params['gamma'] = self.gamma
        elif self.kernel == 'poly':
            params.update({'d': self.d, 'c': self.c})
        return params

    def save_params(self, params_dict):
        filename = f'model_params_{params_dict["kernel"]}.pkl'
        joblib.dump(params_dict, filename)
