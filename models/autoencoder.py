import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent) + '/Repos/datascience/') # Path to datamodel location

import numpy as np
import pandas as pd
from datamodels import datamodels as dm
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Flatten
from keras.metrics import BinaryAccuracy
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

from utils import load_dataset, summarize_results
from models.embedding.embedding import layers_embedding

# References:
# https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798#4214
# https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1
# https://medium.com/analytics-vidhya/building-the-simplest-auto-encoder-in-keras-b7f21f33bef0

# https://www.datacamp.com/tutorial/autoencoder-classifier-python
# https://machinelearningmastery.com/autoencoder-for-classification/
# https://www.geeksforgeeks.org/ml-classifying-data-using-an-auto-encoder/


def main():
    dataset = 'uci'
    batch_size = 16 # 64
    epochs = 30 # 200
    repeats = 10
    lookback_horizon = 48
    prediction_horizon = 1
    embedding = True
    feature_set = 'Light+CO2'
    name = 'autoencoder_embedding'

    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    autoencoders_train = []
    classifiers_train = []
    encoded_representations = []
    for run in range(repeats):
        print('Run ' + str(run + 1))
        set_random_seed(run)
        acc_train, acc_test_1, acc_test_2, acc_test_combined, autoencoder_train, classifier_train, encoded_representation, y_test_combined = run_model(
            dataset, lookback_horizon, prediction_horizon, batch_size, epochs, embedding, feature_set, name)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)
        autoencoders_train.append(autoencoder_train)
        classifiers_train.append(classifier_train)
        encoded_representations.append(encoded_representation)

    max_value = max(scores_test_combined)
    max_value_index = scores_test_combined.index(max_value)

    plot_autoencoder(encoded_representations[max_value_index], y_test_combined, scores_test_combined, name)
    loss_plot(autoencoders_train[max_value_index], name)
    acc_plot(classifiers_train[max_value_index], name)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, name, dataset, batch_size, epochs, repeats, embedding, feature_set)

def run_model(dataset, lookback_horizon, prediction_horizon, batch_size, epochs, embedding, feature_set, name):
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset(dataset=dataset, feature_set=feature_set, normalize=True, embedding=embedding)

    y_shape = y_train.shape[1:]

    # Train autoencoder:

    autoencoder_for_training, autoencoder_for_representation, autoencoder_for_classifier = build_autoencoder(embedding, X_train, y_shape)

    if embedding:
        X_train_target = pd.concat(X_train, axis=1)
    else:
        X_train_target = X_train

    autoencoder_train = autoencoder_for_training.fit(
        X_train, X_train_target,
        batch_size = batch_size,
        epochs = epochs,
        validation_split=0.2,
        shuffle = True,
    )

    # Train classifier:

    for layer in autoencoder_for_classifier.layers:
        if layer.name not in ['classifier_1', 'classifier_2', 'classifier_3', 'classifier_4']:
            layer.trainable = False

    #classifier = build_classifier(autoencoder_for_classifier, y_shape)

    print(autoencoder_for_training.get_weights()[0][1])
    print(autoencoder_for_classifier.get_weights()[0][1])

    classifier_train = autoencoder_for_classifier.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.5,
        shuffle=True,
    )

    pred_train = autoencoder_for_classifier.predict(X_train)
    pred_test_1 = autoencoder_for_classifier.predict(X_test_1)
    pred_test_2 = autoencoder_for_classifier.predict(X_test_2)
    pred_test_combined = autoencoder_for_classifier.predict(X_test_combined)

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

    #representation = build_representation(classifier)

    encoded_representation = autoencoder_for_representation.predict(X_test_combined)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined, autoencoder_train, classifier_train, encoded_representation, y_test_combined

def build_autoencoder(embedding, X_train, target_shape):
    hidden_size = 128
    code_size = 16

    if embedding:
        input_shape = pd.concat(X_train, axis=1).shape
        input_layer, x = layers_embedding(X_train)
    else:
        input_shape = X_train.shape[1:]
        input_layer = Input(shape=input_shape)
        x = input_layer

    def encoder(input_layer):
        encoded = Dense(hidden_size, activation='relu', activity_regularizer = l1(10e-5), name='encoder_1')(input_layer)
        encoded = Dense(hidden_size / 2, activation='relu', activity_regularizer = l1(10e-5), name='encoder_2')(encoded)
        encoded = Dense(hidden_size / 4, activation='relu', activity_regularizer = l1(10e-5), name='encoder_3')(encoded)
        encoded = Dense(code_size, activation='relu', activity_regularizer = l1(10e-5), name='encoder_4')(encoded)
        return encoded

    encoded = encoder(x)

    def decoder(encoded):
        decoded = Dense(hidden_size / 4, activation='relu')(encoded)
        decoded = Dense(hidden_size / 2, activation='relu')(decoded)
        decoded = Dense(hidden_size, activation='relu')(decoded)
        decoded = Flatten()(decoded)
        decoded = Dense(input_shape[-1], activation='relu')(decoded)
        return decoded

    def representation(encoded):
        representation = Flatten(name='classifier_1')(encoded)
        representation = Dense(hidden_size, activation='relu', name='classifier_2')(representation)
        representation = Dense(2, activation='relu', name='classifier_3')(representation) # For plot_autoencoder
        return representation

    representation = representation(encoded)

    def classifier(representation):
        classifier = Dense(units=target_shape[0], activation='sigmoid', name='classifier_4')(representation)
        return classifier

    autoencoder_for_training = Model(inputs=input_layer, outputs=decoder(encoded))
    autoencoder_for_representation = Model(inputs=input_layer, outputs=representation)
    autoencoder_for_classifier = Model(inputs=input_layer, outputs=classifier(representation)) # This one has a seperate output to be used for classification
    
    autoencoder_for_training.compile(loss='mse', optimizer=Adam(), metrics='mse')
    autoencoder_for_representation.compile(loss='mse', optimizer=Adam(), metrics='mse')
    autoencoder_for_classifier.compile(loss='binary_crossentropy', optimizer=Adam(), metrics='binary_accuracy')

    autoencoder_for_training.summary()

    return autoencoder_for_training, autoencoder_for_representation, autoencoder_for_classifier

def build_classifier(autoencoder, input_layer, target_shape):
    hidden_size = 128

    classifier = autoencoder.outputs[0]
    classifier = Flatten()(classifier)
    classifier = Dense(hidden_size, activation='relu')(classifier)
    classifier = Dense(2, activation='relu')(classifier) # For plot_autoencoder
    classifier = Dense(units=target_shape[0], activation='sigmoid')(classifier)

    # Weights of the trained encoder shouldnt be changed
    for layer in classifier.layers[0:5]:
        layer.trainable = False

    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

    classifier.summary()

    return classifier

def build_representation(classifier):
    representation = Sequential(classifier.layers[0:-1])

    return representation

def loss_plot(autoencoder_train, model_name):
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(str(ROOT_DIR) + '/models/results/' + model_name + '_loss_plot.png')
    #plt.show()

def acc_plot(classifier_train, model_name):
    accuracy = classifier_train.history['binary_accuracy']
    val_accuracy = classifier_train.history['val_binary_accuracy']
    epochs = range(len(accuracy))

    plt.figure()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(str(ROOT_DIR) + '/models/results/' + model_name + '_accuracy_plot.png')
    #plt.show()

def plot_autoencoder(encoded_representation, y, scores_test_combined, model_name):
    #tsne = TSNE(n_components=2, random_state=42)

    #X_transformed = tsne.fit_transform(best)

    plt.figure(figsize=(12, 8))

    plt.scatter(
        encoded_representation[np.where(y==0),0],
        encoded_representation[np.where(y==0),1],
        marker='o',
        color='red',
        label='Non-Occupancy'
    )
    plt.scatter(
        encoded_representation[np.where(y==1),0],
        encoded_representation[np.where(y==1),1],
        marker='o',
        color='blue',
        label='Occupancy'
    )
    
    plt.legend(loc='best')
    plt.title('Representation of encoded data')
    plt.xlabel('Output1')
    plt.ylabel('Output2')

    plt.savefig(str(ROOT_DIR) + '/models/results/' + model_name + '_representation.png')
    #plt.show()

if __name__ == '__main__':
    main()
