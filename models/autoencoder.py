import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent) + '/Repos/datascience/') # Path to datamodel location

import numpy as np
import pandas as pd
from datamodels import datamodels as dm
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input
from keras.metrics import BinaryAccuracy
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from utils import load_shaped_dataset, summarize_results

# References:
# https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798#4214
# https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1
# https://medium.com/analytics-vidhya/building-the-simplest-auto-encoder-in-keras-b7f21f33bef0

# https://www.datacamp.com/tutorial/autoencoder-classifier-python
# https://machinelearningmastery.com/autoencoder-for-classification/
# https://www.geeksforgeeks.org/ml-classifying-data-using-an-auto-encoder/


def main():
    batch_size = 16 # 64
    epochs = 3 # 200
    repeats = 2
    lookback_horizon = 48
    prediction_horizon = 1
    name = 'autoencoder'

    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    encodings_test_combined = []
    for run in range(repeats):
        print('Run ' + str(run + 1))
        set_random_seed(42)
        acc_train, acc_test_1, acc_test_2, acc_test_combined, encoded_test_combined, y_test_combined = run_model(lookback_horizon, prediction_horizon, batch_size, epochs, name)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)
        encodings_test_combined.append(encoded_test_combined)

    plot_autoencoder(encodings_test_combined, y_test_combined, scores_test_combined, name)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, name).to_csv('./results/' + name + '.csv')

def run_model(lookback_horizon, prediction_horizon, batch_size, epochs, name):
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_shaped_dataset(lookback_horizon, prediction_horizon, normalize=True)

    x_shape = X_train.shape[1:]
    y_shape = y_train.shape[1:]

    # Train autoencoder:

    autoencoder = build_autoencoder(x_shape)

    autoencoder_train = autoencoder.fit(
        X_train, X_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split=0.2,
        shuffle = True,
    )

    loss_plot(autoencoder_train, name)

    # Train classifier:

    classifier = build_classifier(autoencoder, y_shape)

    print(autoencoder.get_weights()[0][1])
    print(classifier.get_weights()[0][1])

    classifier_train = classifier.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=np.floor(epochs/2),
        validation_split=0.5,
        shuffle=True,
    )

    acc_plot(classifier_train, name)

    encoded_train = classifier.predict(X_train)
    encoded_test_1 = classifier.predict(X_test_1)
    encoded_test_2 = classifier.predict(X_test_2)
    encoded_test_combined = classifier.predict(X_test_combined)

    """svm = SVC()
    svm.fit(X_train, y_train)
    pred_train = svm.predict(X_train)
    pred_test_1 = svm.predict(X_test_1)
    pred_test_2 = svm.predict(X_test_2)
    pred_test_combined = svm.predict(X_test_combined)"""

    acc_train = BinaryAccuracy()(y_train, pred_train)
    acc_test_1 = BinaryAccuracy()(y_test_1, pred_test_1)
    acc_test_2 = BinaryAccuracy()(y_test_2, pred_test_2)
    acc_test_combined = BinaryAccuracy()(y_test_combined, pred_test_combined)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined, encoded_test_combined, y_test_combined

def build_autoencoder(input_shape):
    hidden_size = 128
    code_size = 16

    input_layer = Input(shape=input_shape)

    def encoder(input_layer):
        encoded = Dense(hidden_size, activation='relu', activity_regularizer = l1(10e-5))(input_layer)
        encoded = Dense(hidden_size / 2, activation='relu', activity_regularizer = l1(10e-5))(encoded)
        encoded = Dense(hidden_size / 4, activation='relu', activity_regularizer = l1(10e-5))(encoded)
        encoded = Dense(code_size, activation='relu', activity_regularizer = l1(10e-5))(encoded)
        return encoded

    def decoder(encoded):
        decoded = Dense(hidden_size / 4, activation='relu')(encoded)
        decoded = Dense(hidden_size / 2, activation='relu')(decoded)
        decoded = Dense(hidden_size, activation='relu')(decoded)
        decoded = Dense(input_shape[-1], activation='relu')(decoded)
        return decoded

    autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)))
    
    autoencoder.compile(loss='mse', optimizer=Adam(), metrics='mse')

    autoencoder.summary()

    return autoencoder

def build_classifier(autoencoder, target_shape):
    hidden_size = 128

    classifier = Sequential()
    classifier.add(autoencoder.layers[0])
    classifier.add(autoencoder.layers[1])
    classifier.add(autoencoder.layers[2])
    classifier.add(autoencoder.layers[3])
    classifier.add(autoencoder.layers[4])
    classifier.add(Dense(hidden_size, activation='relu'))
    classifier.add(Dense(units=target_shape[0], activation='sigmoid'))

    # Weights of the trained encoder shouldnt be changed
    for layer in classifier.layers[0:5]:
        layer.trainable = False

    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

    classifier.summary()

    return classifier

def loss_plot(autoencoder_train, model_name):
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./results/' + model_name + '_loss_plot.png')
    #plt.show()

def acc_plot(classifier_train, model_name):
    accuracy = classifier_train.history['acc']
    val_accuracy = classifier_train.history['val_acc']
    epochs = range(len(accuracy))

    plt.figure()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('./results/' + model_name + '_accuracy_plot.png')
    #plt.show()

def plot_autoencoder(encodings_test_combined, y, scores_test_combined, model_name):
    max_value = max(scores_test_combined)
    max_value_index = scores_test_combined.index(max_value)
    best = encodings_test_combined[max_value_index]

    tsne = TSNE(n_components=2, random_state=42)

    X_transformed = tsne.fit_transform(best)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        X_transformed[np.where(y==0),0],
        X_transformed[np.where(y==0),1],
        marker='o',
        color='red',
        label='Non-Occupancy'
    )
    plt.scatter(
        X_transformed[np.where(y==1),0],
        X_transformed[np.where(y==1),1],
        marker='o',
        color='blue',
        label='Occupancy'
    )
    
    plt.legend(loc='best')
    plt.title('TSNE of encoded data')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')

    plt.savefig('./results/' + model_name + '.png')
    #plt.show()

if __name__ == '__main__':
    main()
