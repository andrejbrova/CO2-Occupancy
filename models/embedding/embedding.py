import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.metrics import BinaryAccuracy
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Activation, Concatenate
from keras.layers import Dropout, Dense, Input, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from get_data import load_dataset, get_embeddings, get_embeddings_shaped
from utils import summarize_results
from utils import T2V

# References
# https://medium.com/@jdwittenauer/deep-learning-with-keras-structured-time-series-37a66c6aeb28


def run(
    parameters = {
        'model_name': 'LSTM',
        'dataset': 'uci',
        'feature_set': 'full',
        'split_data': True,
        'epochs': 50,
        'historical_co2': False,
        'embedding': True,
        'windowing': False,
        'runs': 10,
        'batch_size': 32,
        }
    ):

    models = {
        'CNN': layers_CNN,
        'GRU': layers_GRU,
        'LSTM': layers_LSTM,
        'SRNN': layers_SRNN,
        'T2V': layers_T2V, # TODO: Needs to be trained on result
    }

    X_train, X_test_list, y_train, y_test_list, encoders = load_dataset(
        dataset=parameters['dataset'],
        feature_set=parameters['feature_set'],
        historical_co2=parameters['historical_co2'],
        normalize=True,
        embedding=True,
        shaped=parameters['windowing'],
        split_data=parameters['split_data']
        )
    
    models = []
    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(parameters['runs']):
        print('Run: ' + str(run + 1) + ', Dataset: ' + parameters['dataset'] + ', Model: ' + parameters['model_name'])
        
        acc_train, acc_test_1, acc_test_2, acc_test_combined, model = run_model(X_train, X_test_list, y_train, y_test_list, parameters['batch_size'], parameters['epochs'], models[parameters['model_name']], encoders)
        
        models.append(model)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, parameters)#, suffix='_+'+str(historical_co2)+'min')
    
    for cat_var in encoders.keys():
        plot_embedding(models, parameters['dataset'], encoders, cat_var, scores_test_1, parameters['model_name'])        

def run_model(X_train, X_test_list, y_train, y_test_list, batch_size, epochs, model_layers, encoders):
    inputs, x_emb = layers_embedding(X_train, encoders)
    x = model_layers(x_emb)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    #model_emb = Model(inputs=inputs, outputs=x_emb)
    #model_t2v = Model(inputs=inputs, outputs=x_t2v)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
    #model_emb.compile(loss='mse', optimizer='adam', metrics='mse')
    #model_t2v.compile(loss='mse', optimizer='adam', metrics='mse')
    print(model.summary())

    callbacks_list = [
        ReduceLROnPlateau(monitor='binary_accuracy', factor=0.2, patience=3,
                            verbose=1, mode='auto', min_delta=10, cooldown=0,
                            min_lr=0.0001),
        ModelCheckpoint('best_model_weights.hdf5', monitor='binary_accuracy',
                            save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=10)
    ]
    #X_train[0] = np.asarray(X_train[0]).astype('float32')
    #X_train[1] = np.asarray(X_train[1]).astype('float32')
    #y_train = np.asarray(y_train).astype('float32')
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks_list
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test_list = []
    for X_test in X_test_list:
        y_pred_test_list.append(model.predict(X_test))

    acc_train = BinaryAccuracy()(y_train, y_pred_train)
    acc_test_list = []
    for y_test, y_pred_test in zip(y_test_list, y_pred_test_list):
        acc_test_list.append(BinaryAccuracy()(y_test, y_pred_test))

    return acc_train, acc_test_list[0], acc_test_list[1], acc_test_list[2], model # TODO change

def plot_embedding(models, dataset, encoders, category, scores_test_1, model_name):
    # Find best and worst model
    max_value = max(scores_test_1)
    max_value_index = scores_test_1.index(max_value)
    best = models[max_value_index]
    min_value = min(scores_test_1)
    min_value_index = scores_test_1.index(min_value)
    worst = models[min_value_index]
    models_to_plot = [best, worst]
    colors = ['green', 'red']
    labels = ['Best Model', 'Worst Model']

    plt.figure(figsize=(12, 8))

    for it, model in enumerate(models_to_plot):
        embedding_layer = model.get_layer(category)
        weights = embedding_layer.get_weights()[0]
        if weights.shape[-1] <= 1:
            return
        elif weights.shape[-1] > 2:
            pca = PCA(n_components=2)
            weights = pca.fit_transform(weights)
        weights_t = weights.T

        plt.scatter(weights_t[0], weights_t[1], c=colors[it], label=labels[it])
        for i, day in enumerate(encoders[category].classes_):
            plt.annotate(day, (weights_t[0, i], weights_t[1, i]))
            #fig.tight_layout()

    plt.title('PCA of the weights of the embedding "' + category + '" layer')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    if dataset == 'uci':
        folder = 'results/'
    else:
        folder = 'results_' + dataset + '/'

    plt.savefig(str(ROOT_DIR) + '/models/' + folder + model_name + '_embedding_' + category + '.png')
    #plt.show()

# Layers

def layers_embedding(X_train, encoders, time2vec=True):
    input_shape = X_train[0].shape[1:]
    #input_shape[-1] += len(X_train) - 1
    #input_shape = tuple(input_shape)
    cat_vars = encoders.keys()

    #cat_vars = []
    #for x in X_train[0:-1]:
    #    cat_vars.append(x.name)
    #cont_vars = X_train[-1].columns

    # Vector sizes
    cat_sizes = [(c, len(encoders['DayOfWeek'].classes_)) for i, c in enumerate(cat_vars)] # TODO generalize
    embedding_sizes = [(c, min(50, (c + 1) // 2)) for _, c in cat_sizes]

    inputs = []
    embed_layers = []
    for (c, (in_size, out_size)) in zip(cat_vars, embedding_sizes):
        reshape = list(input_shape)
        reshape[-1] = out_size
        reshape = tuple(reshape)
        i = Input(shape=input_shape)
        o = Embedding(in_size, out_size, name=c)(i)
        o = Reshape(target_shape=reshape)(o)
        inputs.append(i)
        embed_layers.append(o)

    embed = Concatenate(axis=-1)(embed_layers)
    embed = Dropout(0.04)(embed)

    cont_input_shape = X_train[-1].shape[1:]
    cont_input = Input(shape=cont_input_shape)
    inputs.append(cont_input)

    x = Concatenate(axis=-1)([embed, cont_input])
    concat_shape = embed.shape[1:].as_list()
    concat_shape[-1] += cont_input.shape[-1]
    #concat_shape.append(1)
    x = keras.layers.Reshape(concat_shape)(x)

    # Time2Vec
    if time2vec:
        x = layers_T2V(x)

    return inputs, x

def layers_CNN(x):
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Dense(units=32, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    return x

def layers_LSTM(x):
    hidden_layer_size = 32

    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LSTM(units=64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=hidden_layer_size, activation='relu')(x)
    
    return x

def layers_GRU(x):
    hidden_layer_size = 32

    x = keras.layers.GRU(return_sequences=True, units=64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.GRU(return_sequences=True, units=64)(x)
    x = keras.layers.GRU(units=64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=hidden_layer_size, activation='relu')(x)

    return x

def layers_SRNN(x): # TODO: Sequence required?
    hidden_layer_size = 32

    #x = keras.layers.Flatten()(x)
    x = keras.layers.SimpleRNN(return_sequences=True, units=64)(x),
    x = keras.layers.SimpleRNN(return_sequences=True, units=64)(x),
    x = keras.layers.Dropout(0.2)(x),
    x = keras.layers.SimpleRNN(return_sequences=True, units=64)(x),
    x = keras.layers.SimpleRNN(return_sequences=True, units=64)(x),
    x = keras.layers.SimpleRNN(units=64)(x),
    x = keras.layers.Dropout(0.2)(x),
    #x = keras.layers.SimpleRNN(units=64)(x),
    x = keras.layers.Dense(units=hidden_layer_size, activation='relu')(x),

    return x

def layers_T2V(x):
    x = T2V(64)(x)
    return x

if __name__ == '__main__':
    run()