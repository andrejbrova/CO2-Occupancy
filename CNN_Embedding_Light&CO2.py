import pathlib
import sys
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
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent) + '/Repos/datascience/')

from utils import get_embeddings, summarize_results
import datamodels as dm


def main():
    batch_size = 32
    epochs = 50
    repeats = 10
    lookback_horizon = 48
    prediction_horizon = 1
    embedding = True
    model_name = 'CNN_embedding'

    run_embedding(batch_size, epochs, repeats, model_name)


def run_embedding(batch_size, epochs, repeats, model_name):
    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, encoders = get_embeddings()

    cat_vars = [
        # 'Week',
        'Weekday'
    ]
    cont_vars = X_train[-1].columns

    # Vector sizes
    cat_sizes = [(c, len(X_train[i].cat.categories)) for i, c in enumerate(cat_vars)]
    embedding_sizes = [(c, min(50, (c + 1) // 2)) for _, c in cat_sizes]

    def CNN_embedding(input_shape: tuple, target_shape: tuple) -> keras.Model:
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
        x = keras.layers.Reshape((6, 1))(x)

        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=2, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation="relu")(x)
        x = keras.layers.Dense(units=32, activation="relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(target_shape[0], activation='sigmoid')(x)

        return Model(inputs=inputs, outputs=x)

    def train_model(model, x_train, y_train) -> keras.callbacks.History:
        callbacks_list = [
            ReduceLROnPlateau(monitor='binary_accuracy', factor=0.2, patience=3,
                              verbose=1, mode='auto', min_delta=10, cooldown=0,
                              min_lr=0.0001),
            ModelCheckpoint('best_model_weights.hdf5', monitor='binary_accuracy',
                            save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='binary_accuracy', patience=10)
        ]
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks_list
        )

    def run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, model):
        y_pred_train = model.predict(X_train)
        y_pred_test_1 = model.predict(X_test_1)
        y_pred_test_2 = model.predict(X_test_2)
        y_pred_test_combined = model.predict(X_test_combined)

        evaluation = BinaryAccuracy()
        acc_train = evaluation(y_train, y_pred_train)
        evaluation.reset_states()
        acc_test_1 = evaluation(y_test_1, y_pred_test_1)
        evaluation.reset_states()
        acc_test_2 = evaluation(y_test_2, y_pred_test_2)
        evaluation.reset_states()
        acc_test_combined = evaluation(y_test_combined, y_pred_test_combined)

        return acc_train, acc_test_1, acc_test_2, acc_test_combined

    models = []
    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(repeats):
        print('Run ' + str(run + 1))

        model = CNN_embedding((1,), (1,))
        optimizer = keras.optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
        print(model.summary())

        train_model(model, X_train, y_train)
        acc_train, acc_test_1, acc_test_2, acc_test_combined = run_model(X_train, X_test_1, X_test_2, X_test_combined,
                                                                         y_train, y_test_1, y_test_2, y_test_combined,
                                                                         model)

        models.append(model)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)
    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, model_name).to_csv(
        str(directory) + '/results/' + model_name + '.csv')

    max_value = max(scores_test_combined)
    max_value_index = scores_test_combined.index(max_value)
    plot_embedding(models[max_value_index], encoders, 'Weekday', model_name)


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
    # plt.show()


if __name__ == '__main__':
    main()