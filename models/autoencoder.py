import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent) + '/Repos/datascience/') # Path to datamodel location

import numpy as np
import pandas as pd
from datamodels import datamodels as dm
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Input, Flatten, Reshape
from keras.metrics import BinaryAccuracy
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

from get_data import load_dataset
from utils import summarize_results
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
    epochs = 30
    repeats = 10
    historical_co2 = False
    embedding = True
    feature_set = 'Light+CO2'
    name = 'autoencoder'
    shaped = True
    split_data = True
    plot_representation = True
    code_size = 2 # 2 for plotting, 6 for brick dataset


    X_train, X_test_list, y_train, y_test_list, encoders = load_dataset(dataset=dataset, feature_set=feature_set, normalize=True, embedding=embedding, historical_co2=historical_co2, shaped=shaped)

    scores_train = []
    scores_test_list = []
    autoencoders_train = []
    classifiers_train = []
    encoded_representations = []
    predictions_list = []
    for run in range(repeats):
        print('Run: ' + str(run + 1) + ', Dataset: ' + dataset + ', Model: ' + name)
        set_random_seed(run)
        acc_train, acc_test_list, autoencoder_train, classifier_train, encoded_representation, y_pred_test_list = run_model(
            X_train, X_test_list, y_train, y_test_list, dataset, batch_size, epochs, embedding, historical_co2, feature_set, code_size, encoders, name)
        scores_train.append(acc_train)
        scores_test_list.append(acc_test_list)
        autoencoders_train.append(autoencoder_train)
        classifiers_train.append(classifier_train)
        encoded_representations.append(encoded_representation)
        predictions_list.append(y_pred_test_list)

    if plot_representation:
        scores_test = [sublist[-1] for sublist in scores_test_list] # Gets test combined if datasetis uci
        max_value = max(scores_test)
        max_value_index = scores_test.index(max_value)

        plot_autoencoder(encoded_representations[max_value_index], predictions_list[max_value_index], y_test_list, scores_test, name)
        loss_plot(autoencoders_train[max_value_index], name)
        acc_plot(classifiers_train[max_value_index], name)
        plot_densities(dataset, feature_set, historical_co2, predictions_list[max_value_index], name)

    summarize_results(scores_train, scores_test_list, name, dataset, batch_size, epochs, repeats, embedding, feature_set, historical_co2)#, suffix='_+'+str(historical_co2)+'min')

def run_model(X_train, X_test_list, y_train, y_test_list, dataset, batch_size, epochs, embedding, historical_co2, feature_set, code_size, encoders, name):
    y_shape = y_train.shape[1:]

    # Train autoencoder:

    autoencoder_for_training, autoencoder_for_representation, autoencoder_for_classifier = build_autoencoder(embedding, X_train, y_shape, code_size, encoders)

    if embedding:
        if type(X_train[0]) == np.ndarray:
            X_train_target = np.concatenate(X_train, axis=-1)
        else:
            X_train_target = pd.concat(X_train, axis=1)
    else:
        X_train_target = X_train

    callbacks_list_training = [
        EarlyStopping(monitor='val_mse', patience=10)
    ]

    autoencoder_train = autoencoder_for_training.fit(
        X_train, X_train_target,
        batch_size = batch_size,
        epochs = epochs,
        validation_split=0.2,
        shuffle = True,
        callbacks = callbacks_list_training,
    )

    # Train classifier:

    for layer in autoencoder_for_classifier.layers:
        if layer.name not in ['classifier_1', 'classifier_2', 'classifier_3', 'classifier_4']:
            layer.trainable = False

    if not np.array_equal(autoencoder_for_training.get_weights()[0], autoencoder_for_classifier.get_weights()[0]):
        raise Exception("Weights of both encoders have to be identical")

    callbacks_list_classification = [
        EarlyStopping(monitor='val_binary_accuracy', patience=10)
    ]

    classifier_train = autoencoder_for_classifier.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.5,
        shuffle=True,
        callbacks = callbacks_list_classification
    )

    pred_train = autoencoder_for_classifier.predict(X_train)
    pred_test_list = []
    for X_test in X_test_list:
        pred_test_list.append(autoencoder_for_classifier.predict(X_test))

    """svm = SVC()
    svm.fit(X_train, y_train)
    pred_train = svm.predict(X_train)
    pred_test_1 = svm.predict(X_test_1)
    pred_test_2 = svm.predict(X_test_2)
    pred_test_combined = svm.predict(X_test_combined)"""

    acc_train = BinaryAccuracy()(y_train, pred_train)
    acc_test_list = []
    for y_test, pred_test in zip(y_test_list, pred_test_list):
        acc_test_list.append(BinaryAccuracy()(y_test, pred_test))

    encoded_representation = []
    for X_test in X_test_list:
        encoded_representation.append(autoencoder_for_representation.predict(X_test))

    return acc_train, acc_test_list, autoencoder_train, classifier_train, encoded_representation, pred_test_list

def build_autoencoder(embedding, X_train, target_shape, code_size, encoders):
    hidden_size = 128

    if embedding:
        if type(X_train[0]) == np.ndarray:
            input_shape = np.concatenate(X_train, axis=-1).shape[1:]
        else:
            input_shape = pd.concat(X_train, axis=1).shape[1:]
        input_layer, x = layers_embedding(X_train, encoders)
    else:
        input_shape = X_train.shape[1:]
        input_layer = Input(shape=input_shape)
        x = input_layer

    def encoder(input_layer):
        encoded = Flatten()(input_layer)
        encoded = Dense(hidden_size, activation='relu', activity_regularizer = l1(10e-5), name='encoder_1')(encoded)
        encoded = Dense(hidden_size / 4, activation='relu', activity_regularizer = l1(10e-5), name='encoder_2')(encoded)
        encoded = Dense(hidden_size / 8, activation='relu', activity_regularizer = l1(10e-5), name='encoder_3')(encoded)
        encoded = Dense(code_size, activation='relu', activity_regularizer = l1(10e-5), name='encoder_4')(encoded)
        return encoded

    encoded = encoder(x)

    def decoder(encoded):
        decoded = Dense(hidden_size / 8, activation='relu')(encoded)
        decoded = Dense(hidden_size / 4, activation='relu')(decoded)
        decoded = Dense(hidden_size, activation='relu')(decoded)
        decoded = Dense(np.prod(list(input_shape)), activation='relu')(decoded)
        decoded = Reshape(input_shape)(decoded)
        #decoded = Flatten()(decoded)
        #decoded = Dense(input_shape[-1], activation='relu')(decoded)
        return decoded

    def classifier(encoded):
        classifier = Flatten(name='classifier_1')(encoded)
        classifier = Dense(hidden_size / 2, activation='relu', name='classifier_2')(classifier)
        #classifier = Dense(2, activation='relu', name='classifier_3')(classifier) # For plot_autoencoder
        classifier = Dense(units=target_shape[0], activation='sigmoid', name='classifier_4')(classifier)
        return classifier

    autoencoder_for_training = Model(inputs=input_layer, outputs=decoder(encoded))
    autoencoder_for_representation = Model(inputs=input_layer, outputs=encoded)
    autoencoder_for_classifier = Model(inputs=input_layer, outputs=classifier(encoded)) # This one has a seperate output to be used for classification
    
    autoencoder_for_training.compile(loss='mse', optimizer=Adam(), metrics='mse')
    autoencoder_for_representation.compile(loss='mse', optimizer=Adam(), metrics='mse')
    autoencoder_for_classifier.compile(loss='binary_crossentropy', optimizer=Adam(), metrics='binary_accuracy')

    autoencoder_for_training.summary()

    return autoencoder_for_training, autoencoder_for_representation, autoencoder_for_classifier

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

def plot_autoencoder(encoded_representations, y_pred_list, y_test_list, scores_test_combined, model_name):
    #tsne = TSNE(n_components=2, random_state=42)

    #X_transformed = tsne.fit_transform(best)

    names = ['1', '2']

    for y, y_pred, name, encoded_representation in zip(y_test_list, y_pred_list, names, encoded_representations):
        y = y.to_numpy()[:,0]
        y_pred = y_pred[:,0].round()

        plt.figure(figsize=(12, 8))

        condition = np.all([y==1, y_pred==1], axis=0)
        number_instances = np.count_nonzero(condition)
        plt.scatter(
            encoded_representation[condition,0],
            encoded_representation[condition,1],
            marker='o',
            color='blue',
            label='True Occupancy (' + str(number_instances) + ' instances, ' + str(round(((number_instances / len(y)) * 100), 1)) + '%)'
        )
        condition = np.all([y==1, y_pred==0], axis=0)
        number_instances = np.count_nonzero(condition)
        plt.scatter(
            encoded_representation[condition,0],
            encoded_representation[condition,1],
            marker='o',
            color='lightblue',
            label='False Non-Occupancy (' + str(number_instances) + ' instances, ' + str(round(((number_instances / len(y)) * 100), 1)) + '%)'
        )
        condition = np.all([y==0, y_pred==0], axis=0)
        number_instances = np.count_nonzero(condition)
        plt.scatter(
            encoded_representation[condition,0],
            encoded_representation[condition,1],
            marker='o',
            color='red',
            label='True Non-Occupancy (' + str(number_instances) + ' instances, ' + str(round(((number_instances / len(y)) * 100), 1)) + '%)'
        )
        condition = np.all([y==0, y_pred==1], axis=0)
        number_instances = np.count_nonzero(condition)
        plt.scatter(
            encoded_representation[condition,0],
            encoded_representation[condition,1],
            marker='o',
            color='salmon',
            label='False Occupancy (' + str(number_instances) + ' instances, ' + str(round(((number_instances / len(y)) * 100), 1)) + '%)'
        )
        
        plt.legend(loc='best')
        plt.title('Representation of encoded data (test ' + name + ')')
        plt.xlabel('Output1')
        plt.ylabel('Output2')

        plt.savefig(str(ROOT_DIR) + '/models/results/' + model_name + '_representation_test_' + name + '.png')
        #plt.show()

def plot_densities(dataset, feature_set, historical_co2, y_pred_test_list, model_name):
    X_train, X_test_list, y_train, y_test_list = load_dataset(dataset=dataset, feature_set=feature_set, normalize=False, embedding=False, historical_co2=historical_co2, shaped=False)
    conditions = [(1,1), (1,0), (0,0), (0,1)]
    condition_names = ['True Occupant', 'False Non-Occupant', 'True Non-Occupant', 'False Occupant']
    export_suffix = ['TP', 'FN', 'TN', 'FP']

    X_test_1 = X_test_list[0] # Will only work for uci dataset
    X_test_2 = X_test_list[1]
    y_test_1 = y_test_list[0]
    y_test_2 = y_test_list[1]
    y_pred_test_1 = y_pred_test_list[0].round()
    y_pred_test_2 = y_pred_test_list[1].round()

    features = X_test_list[0].select_dtypes(exclude=['category', 'string', 'object']).columns
    fig, axs = plt.subplots(len(features), 4, figsize=(4*4, 3*len(features)))
    fig.subplots_adjust(hspace=0.3)
    
    for row, feature in enumerate(features):
        for col in range(4):
            condition_1 = np.all([y_test_1==conditions[col][0], y_pred_test_1==conditions[col][1]], axis=0)
            condition_2 = np.all([y_test_2==conditions[col][0], y_pred_test_2==conditions[col][1]], axis=0)

            if np.count_nonzero(condition_1) > 1:
                X_test_1.loc[condition_1, feature].plot.kde(ax = axs[row, col], label='Test 1')
            if np.count_nonzero(condition_2) > 1:
                X_test_2.loc[condition_2, feature].plot.kde(ax = axs[row, col], label='Test 2')

            axs[row, col].legend()

        plt.setp(axs[row, 0], ylabel=feature)

    for col in range(4):
        plt.setp(axs[-1, col], xlabel=condition_names[col])

    plt.suptitle('Density for ' + model_name + ' data points')
    plt.tight_layout()
    plt.savefig(str(ROOT_DIR) + '/models/results/' + model_name + '_densities.png')
    #plt.show()

if __name__ == '__main__':
    main()
