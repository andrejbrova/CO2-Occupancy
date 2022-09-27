import pathlib
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import layers
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam

directory = pathlib.Path(__file__).parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent) + '/Repos/datascience/') # Path to datamodel location

from utils import load_shaped_dataset, summarize_results
from datamodels import datamodels as dm

# References:
# https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/
# https://github.com/yasharvindsingh/ResNet20


def main():
    batch_size = 32
    epochs = 100
    repeats = 10
    lookback_horizon = 48
    prediction_horizon = 1
    n = 3
    name = 'resnet20'

    depth = n * 6 + 2

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
        acc_train, acc_test_1, acc_test_2, acc_test_combined = run_model(lookback_horizon, prediction_horizon, batch_size, epochs, depth, name)
        scores_train.append(acc_train)
        scores_test_1.append(acc_test_1)
        scores_test_2.append(acc_test_2)
        scores_test_combined.append(acc_test_combined)

    summarize_results(scores_train, scores_test_1, scores_test_2, scores_test_combined, name).to_csv(str(directory.parent) + '/results/' + name + '.csv')

def build_model(input_shape, target_shape, depth):

    def resnet_layer(
        inputs,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation='relu',
        batch_normalization=True,
        conv_first=True
        ):
        
        conv = layers.Conv1D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4)
        )

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = layers.Activation(activation)(x)
        else:
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = layers.Activation(activation)(x)
            x = conv(x)

        return x

    def resnet_v1(input_shape, target_shape, depth):
  
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)
    
        inputs = layers.Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    strides=strides
                )
                y = resnet_layer(
                    inputs=y,
                    num_filters=num_filters,
                    activation=None
                )
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False
                    )
                x = layers.add([x, y])
                x = layers.Activation('relu')(x)
            num_filters *= 2
    
        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = layers.AveragePooling1D(pool_size=8)(x)
        y = layers.Flatten()(x)
        outputs = layers.Dense(
            target_shape[0],
            activation='sigmoid',
            kernel_initializer='he_normal')(y)
    
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

        return model
    
    def compile_model(model: Model):
        optimizer = Adam(learning_rate = lr_schedule(0))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

    model = resnet_v1(input_shape=input_shape, target_shape=target_shape, depth=depth)
    compile_model(model)
    model.summary()

    return model

def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)

        return lr

def run_model(lookback_horizon, prediction_horizon, batch_size, epochs, depth, name):
    

    X_train, X_test_1, X_test_2, X_test_combined, y_train, y_test_1, y_test_2, y_test_combined = load_shaped_dataset(lookback_horizon, prediction_horizon)

    x_scaler = dm.processing.Normalizer().fit(X_train)
    
    x_shape = X_train.shape[1:]
    y_shape = y_train.shape[1:]

    model = build_model(x_shape, y_shape, depth)

    X_train = x_scaler.transform(X_train)
    X_test_1 = x_scaler.transform(X_test_1)
    X_test_2 = x_scaler.transform(X_test_2)
    X_test_combined = x_scaler.transform(X_test_combined)

    def train_model(model, x_train, y_train):# -> keras.callbacks.History:  
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1),
                                    cooldown = 0,
                                    patience = 5,
                                    min_lr = 0.5e-6)
        
        callbacks = [lr_reducer, lr_scheduler]
        
        return model.fit(
            x_train, y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_split=0.2,
            shuffle = True,
            callbacks = callbacks
        )

    train_model(model, X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test_1 = model.predict(X_test_1)
    y_pred_test_2 = model.predict(X_test_2)
    y_pred_test_combined = model.predict(X_test_combined)

    acc_train = BinaryAccuracy()(y_train, y_pred_train)
    acc_test_1 = BinaryAccuracy()(y_test_1, y_pred_test_1)
    acc_test_2 = BinaryAccuracy()(y_test_2, y_pred_test_2)
    acc_test_combined = BinaryAccuracy()(y_test_combined, y_pred_test_combined)

    return acc_train, acc_test_1, acc_test_2, acc_test_combined

if __name__ == '__main__':
    main()