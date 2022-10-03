from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent) + '/Repos/datascience/') # Path to datamodel location

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.metrics import BinaryAccuracy
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate
from keras.layers import Dropout, Dense, Input, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils import load_dataset, summarize_results
from models.embedding.embedding import layers_embedding
from datamodels import datamodels as dm


def main():
    dataset = 'uci'
    batch_size = 32
    epochs = 50
    repeats = 10
    historical_co2 = True
    embedding = False # Wont work here
    feature_set = 'CO2'
    model_name = 'SRNN'

    models = {
        'CNN': (dm.ConvolutionNetwork, layers_CNN),
        'SRNN': (dm.RecurrentNetwork, layers_SRNN)
    }

    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset(
        dataset=dataset, feature_set=feature_set, historical_co2=historical_co2, normalize=True, embedding=embedding, shaped=True)

    """print(f'Training on {X_train.shape[0]} samples.')
    print(f'Testing on {X_test_1.shape[0]} samples (Test1).')
    print(f'Testing on {X_test_2.shape[0]} samples (Test2).')
    print(f'input: {X_train.shape[-1]} features ({X_train.columns.tolist()}).')"""

    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(repeats):
        print('Run ' + str(run + 1))
        model = build_model(X_train.shape[0], X_train.shape[-1], 1, batch_size, epochs, models[model_name], model_name)
        acc_train, acc_test_1, acc_test_2, acc_test_combined = run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, model)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)
    
    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, model_name, dataset, batch_size, epochs, repeats, embedding, feature_set, historical_co2)

def build_model(n_timesteps, n_features, target_shape, batch_size, epochs, model_type, name):
    
    def compile_model(model: keras.Model):
        optimizer = keras.optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

    def train_model(model, x_train, y_train) -> keras.callbacks.History:
        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='binary_accuracy', patience=10)
        ]
        return model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks_list
        )

    model = model_type[0](
        name='SIMPLE ' + name,
        y_scaler_class=dm.processing.Normalizer,
        compile_function=compile_model,
        build_function=model_type[1],
        train_function=train_model)

    return model

def run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, model):
    model.train(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test_1 = model.predict(X_test_1)
    y_pred_test_2 = model.predict(X_test_2)
    y_pred_test_combined = model.predict(X_test_combined)

    acc_train = BinaryAccuracy()(y_train, y_pred_train)
    acc_test_1 = BinaryAccuracy()(y_test_1, y_pred_test_1)
    acc_test_2 = BinaryAccuracy()(y_test_2, y_pred_test_2)
    acc_test_combined = BinaryAccuracy()(y_test_combined, y_pred_test_combined)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined

# Layers

def layers_CNN(input_shape: tuple, target_shape: tuple) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.Dense(units=32, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(target_shape[0], activation='sigmoid')
    ])

def layers_SRNN(input_shape: tuple, target_shape: tuple) -> keras.Model:
    hidden_layer_size = 32
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.SimpleRNN(return_sequences=True, units=64),
            keras.layers.SimpleRNN(return_sequences=True, units=64),
            keras.layers.Dropout(0.2),
            keras.layers.SimpleRNN(return_sequences=True, units=64),
            keras.layers.SimpleRNN(return_sequences=True, units=64),
            keras.layers.SimpleRNN(units=64),
            keras.layers.Dropout(0.2),
            #keras.layers.SimpleRNN(units=64),
            keras.layers.Dense(units=hidden_layer_size, activation='relu'),
            keras.layers.Dense(units=target_shape[0], activation='sigmoid')
        ]
    )

if __name__ == '__main__':
    main()