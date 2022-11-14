import pathlib
import sys
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.metrics import BinaryAccuracy
from sklearn.metrics import accuracy_score

directory = pathlib.Path(__file__).parent
sys.path.append('C:/Users/Brova/Downloads/Smart2B-main/Smart2B-main')
import datamodels as dm

from utils import load_dataset, summarize_results


def main():
    BATCH_SIZE = 32
    EPOCHS = 50
    REPEATS = 10
    lookback_horizon = 48
    prediction_horizon = 1

    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_dataset()

    print(f'Training on {X_train.shape[0]} samples.')
    print(f'Testing on {X_test_1.shape[0]} samples (Test1).')
    print(f'Testing on {X_test_2.shape[0]} samples (Test2).')
    print(f'input: {X_train.shape[-1]} features ({X_train.columns.tolist()}).')

    scores_train = []
    scores_test_1 = []
    scores_test_2 = []
    scores_test_combined = []
    for run in range(REPEATS):
        print('Run ' + str(run + 1))
        model = build_model(X_train.shape[0], X_train.shape[1], 1, BATCH_SIZE, EPOCHS)
        acc_train, acc_test_1, acc_test_2, acc_test_combined = run_model(X_train, X_test_1, X_test_2, X_test_combined,
                                                                         y_train, y_test_1, y_test_2, y_test_combined,
                                                                         model, lookback_horizon, prediction_horizon)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)
    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, 'GRU')


def build_model(n_timesteps, n_features, target_shape, batch_size, epochs):
    def GRU_build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
        hidden_layer_size = 32
        return keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    keras.layers.GRU(return_sequences=True, units=64),
                    keras.layers.Dropout(0.2),
                    keras.layers.GRU(return_sequences=True, units=64),
                    keras.layers.GRU(units=64),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(units=hidden_layer_size, activation='relu'),
                    keras.layers.Dense(units=target_shape[0])
                ]
            )

    def compile_model(model: keras.Model):
        optimizer = keras.optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

    def train_model(model, x_train, y_train) -> keras.callbacks.History:
        callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='binary_accuracy', patience=1000)
        ]
        return model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks_list
        )

    model = dm.GRU(
        name='GRU',
        y_scaler_class=dm.processing.Normalizer,
        compile_function=compile_model,
        build_function=GRU_build_model,
        train_function=train_model)

    return model


def run_model(X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined, model, lookback_horizon, prediction_horizon):
    x_scaler = dm.processing.Normalizer().fit(X_train)
    #y_scaler = dm.processing.Normalizer().fit(y_train)

    model.set_x_scaler(x_scaler)
    #model.set_y_scaler(y_scaler) # Not necessary for binary classification

    X_train, y_train = dm.processing.shape.get_windows(
        lookback_horizon, X_train.to_numpy(), prediction_horizon, y_train.to_numpy()
    )

    X_test_1, y_test_1 = dm.processing.shape.get_windows(
        lookback_horizon, X_test_1.to_numpy(), prediction_horizon, y_test_1.to_numpy(),
    )

    X_test_2, y_test_2 = dm.processing.shape.get_windows(
        lookback_horizon, X_test_2.to_numpy(), prediction_horizon, y_test_2.to_numpy(),
    )

    X_test_combined, y_test_combined = dm.processing.shape.get_windows(
        lookback_horizon, X_test_combined.to_numpy(), prediction_horizon, y_test_combined.to_numpy(),
    )

    model.train(X_train, y_train)

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


if __name__ == '__main__':
    main()
