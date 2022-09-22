import pathlib
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent) + '/Repos/datascience/') # Path to datamodel location

from utils import load_features, summarize_results
from datamodels import datamodels as dm

# References:
# https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798#4214
# https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1
# https://medium.com/analytics-vidhya/building-the-simplest-auto-encoder-in-keras-b7f21f33bef0


def main():
    batch_size = 32
    epochs = 30
    repeats = 2
    lookback_horizon = 48
    prediction_horizon = 1
    name = 'autoencoder'

    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(repeats):
        print('Run ' + str(run + 1))
        acc_train, acc_test_1, acc_test_2, acc_test_combined = run_model(lookback_horizon, prediction_horizon, batch_size, epochs, name)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, name).to_csv(str(directory.parent) + '/results/' + name + '.csv')

def build_model(input_shape):

    def autoencoder(input_shape):
        hidden_size = 128
        code_size = 32

        x = Input(shape=input_shape)
        encoder = Dense(hidden_size, activation='relu')(x)
        code = Dense(code_size, activation='relu')(encoder)
        decoder = Dense(hidden_size, activation='relu')(code)
        y = Dense(input_shape[1], activation='sigmoid')(decoder)

        model = Model(inputs=x, outputs=y)

        return model
    
    def compile_model(model: Model):
        optimizer = Adam()
        model.compile(loss='mse', optimizer='adam', metrics='binary_accuracy')

    model = autoencoder(input_shape=input_shape)
    compile_model(model)
    model.summary()

    return model

def run_model(lookback_horizon, prediction_horizon, batch_size, epochs, name):
    training, test1, test2 = load_features()
    test_combined = pd.concat([test1, test2])

    training = dm.processing.shape.get_windows(
        lookback_horizon, training.to_numpy(), prediction_horizon
    )

    test1 = dm.processing.shape.get_windows(
        lookback_horizon, test1.to_numpy(), prediction_horizon
    )

    test2 = dm.processing.shape.get_windows(
        lookback_horizon, test2.to_numpy(), prediction_horizon
    )

    test_combined = dm.processing.shape.get_windows(
        lookback_horizon, test_combined.to_numpy(), prediction_horizon
    )

    shape = training.shape[1:]

    model = build_model(shape)

    x_scaler = dm.processing.Normalizer().fit(training)

    training = x_scaler.transform(training)
    test1 = x_scaler.transform(test1)
    test2 = x_scaler.transform(test2)
    test_combined = x_scaler.transform(test_combined)

    def train_model(model, x_train):# -> keras.callbacks.History:  
        
        return model.fit(
            x_train, x_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_split=0.2,
            shuffle = True,
        )

    train_model(model, training)

    pred_train = model.predict(training)
    pred_test_1 = model.predict(test1)
    pred_test_2 = model.predict(test2)
    pred_test_combined = model.predict(test_combined)

    acc_train = model.evaluate(training, pred_train)
    acc_test_1 = model.evaluate(test1, pred_test_1)
    acc_test_2 = model.evaluate(test2, pred_test_2)
    acc_test_combined = model.evaluate(test_combined, pred_test_combined)

    plot_embedding(model, encoders, 'Weekday', model_name)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined

def plot_embedding(model, encoders, category, model_name):
    embedding_layer = model.get_layer(category)
    weights = embedding_layer.get_weights()[0]
    pca = PCA(n_components=2)
    weights = pca.fit_transform(weights)
    weights_t = weights.T
    fig, ax = plt.subplots(figsize=(8, 8 * 3 / 4))
    ax.scatter(weights_t[0], weights_t[1])
    for i, day in enumerate(encoders[category].classes_):
        ax.annotate(day, (weights_t[0, i], weights_t[1, i]))
        fig.tight_layout()

    plt.savefig(str(directory) + '/results/' + model_name + '.png')
    #plt.show()

if __name__ == '__main__':
    main()