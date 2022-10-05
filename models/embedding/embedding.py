import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

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

from utils import load_dataset, get_embeddings, summarize_results


def main():
    dataset = 'Italy'
    feature_set='full'
    historical_co2=False
    batch_size = 32
    epochs = 2
    repeats = 2
    model_name = 'LSTM'

    models = {
        'CNN': layers_CNN,
        'GRU': layers_GRU,
        'LSTM': layers_LSTM
    }

    run_embedding(dataset, feature_set, historical_co2, batch_size, epochs, repeats, models[model_name], model_name)

def run_embedding(dataset, feature_set, historical_co2, batch_size, epochs, repeats, model_layers, model_name):
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset(
        dataset=dataset,
        feature_set=feature_set,
        historical_co2=historical_co2,
        normalize=True,
        shaped=False
    )
    X_train, X_test_1, X_test_2, X_test_combined, encoders = get_embeddings(
        X_train, X_test_1, X_test_2, X_test_combined)
    
    models = []
    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(repeats):
        print('Run: ' + str(run + 1) + ', Dataset: ' + dataset + ', Model: ' + model_name)
        
        acc_train, acc_test_1, acc_test_2, acc_test_combined, model = run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, batch_size, epochs, model_layers)
        
        models.append(model)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, model_name, dataset, batch_size, epochs, repeats, True, feature_set, historical_co2)
    
    for cat_var in encoders.keys():
        plot_embedding(models, dataset, encoders, cat_var, scores_test_1, model_name)        

def run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, batch_size, epochs, model_layers):
    inputs, x = layers_embedding(X_train)
    x = model_layers(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
    print(model.summary())

    callbacks_list = [
        ReduceLROnPlateau(monitor='binary_accuracy', factor=0.2, patience=3,
                            verbose=1, mode='auto', min_delta=10, cooldown=0,
                            min_lr=0.0001),
        ModelCheckpoint('best_model_weights.hdf5', monitor='binary_accuracy',
                            save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=10)
    ]

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks_list
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test_1 = model.predict(X_test_1)
    y_pred_test_2 = model.predict(X_test_2)
    y_pred_test_combined = model.predict(X_test_combined)

    acc_train = BinaryAccuracy()(y_train, y_pred_train)
    acc_test_1 = BinaryAccuracy()(y_test_1, y_pred_test_1)
    acc_test_2 = BinaryAccuracy()(y_test_2, y_pred_test_2)
    acc_test_combined = BinaryAccuracy()(y_test_combined, y_pred_test_combined)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined, model

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

def layers_embedding(X_train):
    input_shape = (1,)

    cat_vars = []
    for x in X_train[0:-1]:
        cat_vars.append(x.name)
    cont_vars = X_train[-1].columns

    # Vector sizes
    cat_sizes = [(c, len(X_train[i].cat.categories)) for i, c in enumerate(cat_vars)]
    embedding_sizes = [(c, min(50, (c + 1) // 2)) for _, c in cat_sizes]

    inputs = []
    embed_layers = []
    for (c, (in_size, out_size)) in zip(cat_vars, embedding_sizes):
        i = Input(shape=input_shape)
        o = Embedding(in_size, out_size, name=c)(i)
        o = Reshape(target_shape=(out_size,))(o)
        inputs.append(i)
        embed_layers.append(o)

    embed = Concatenate()(embed_layers)
    embed = Dropout(0.04)(embed)

    cont_input = Input(shape=(len(cont_vars),))
    inputs.append(cont_input)

    x = Concatenate()([embed, cont_input])
    concat_shape = (embed.shape[1] + cont_input.shape[1], 1)
    x = keras.layers.Reshape(concat_shape)(x)

    return inputs, x

def layers_CNN(x):
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    #x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
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

if __name__ == '__main__':
    main()