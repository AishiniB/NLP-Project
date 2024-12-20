'''This code file helps to train a CNN classifier based on the feature set obtained after running
feature_building.py.
Input: Feature set obtained on running feature_building.py, merged dataset obtained from merger.py
Output: Classification results'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

from training import StockTrendPredictor


class CNNModel:
    def __init__(self, input_shape, num_classes=3, params=None):
        # Define hyperparameters
        if params is None:
            params = {
                'conv1_filters': 64,
                'conv2_filters': 128,
                'conv3_filters': 256,
                'dense_units': 256,
                'dropout_rate': 0.6,
                'learning_rate': 0.001,
            }
        self.params = params

        # Build model
        self.model = Sequential([
            Conv1D(params['conv1_filters'], kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Conv1D(params['conv2_filters'], kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Conv1D(params['conv3_filters'], kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Flatten(),
            Dense(params['dense_units'], activation='relu'),
            Dropout(params['dropout_rate']),
            Dense(num_classes, activation='softmax'),
        ])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate on test data
        y_pred = self.model.predict(X_test).argmax(axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, history

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix - CNN'):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Downward', 'Neutral', 'Upward'],
                    yticklabels=['Downward', 'Neutral', 'Upward'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


def main():
    # Initialize predictor for CNN
    predictor = StockTrendPredictor()
    X, y = predictor.load_data('News+Stock data1.csv', 'final_features1.csv')

    # Ensure X can be reshaped for CNN
    if X is None or y is None:
        print("Failed to load data. Exiting...")
        return

    # Reshape data for CNN
    X_reshaped = np.expand_dims(X.values, axis=2)  # Reshape for Conv1D
    input_shape = X_reshaped.shape[1:]  # (timesteps, features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define hyperparameter grid
    param_grid = {
        'conv1_filters': [32, 64],
        'conv2_filters': [64, 128],
        'conv3_filters': [128, 256],
        'dense_units': [128, 256],
        'dropout_rate': [0.4, 0.6],
        'learning_rate': [0.001, 0.0005],
    }

    # Perform grid search
    best_params = None
    best_accuracy = 0

    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")

        # Initialize and train CNN
        cnn_model = CNNModel(input_shape=input_shape, num_classes=3, params=param_dict)
        accuracy, _ = cnn_model.train_and_evaluate(X_train, X_test, y_train, y_test)

        print(f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = param_dict

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()

