import sys
from datetime import datetime
from pathlib import Path

from pandas import DatetimeIndex

ROOT_DIR = str(Path(__file__).parents[0])
sys.path.append(ROOT_DIR)

import os
import glob
import numpy as np
import pandas as pd
from keras.layers import Layer
from tensorflow.keras import backend as K


def summarize_results(
        scores_train,
        scores_test_list,
        model_name='?',
        dataset='?',
        batch_size='?',
        epochs='?',
        repeats='?',
        embedding='?',
        feature_set='?',
        historical_co2='?',
        suffix=''
):
    result = {
        'batch size': batch_size,
        'epochs': epochs,
        'repeats': repeats,
        'embedding': embedding,
        'feature set': feature_set,
        'historical co2': historical_co2,
        'accuracy_train_mean': np.mean(scores_train),
        'accuracy_train_std': np.std(scores_train),
        'accuracy_test_1_mean': np.mean([sublist[0] for sublist in scores_test_list]),
        'accuracy_test_1_std': np.std([sublist[0] for sublist in scores_test_list]),
    }

    if feature_set != '?' and feature_set != 'full':
        suffix += '_' + feature_set
    if embedding != '?' and embedding != False:
        suffix += '_embedding'
    if dataset == '?' or dataset == 'uci':
        result.update({  # Add evaluation for second and combined test set
            'accuracy_test_2_mean': np.mean([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_2_std': np.std([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_combined_mean': np.mean([sublist[2] for sublist in scores_test_list]),
            'accuracy_test_combined_std': np.std([sublist[2] for sublist in scores_test_list])
        })
        folder = 'results/'
    else:
        folder = 'results_' + dataset + '/'

    pd.DataFrame(result, index=[model_name]).to_csv(ROOT_DIR + '/models/' + folder + model_name + suffix + '.csv')


def concat_tables():
    path = ROOT_DIR + '/models/results/'
    csv_files = glob.glob(os.path.join(path, "*.csv"))

    tables = []
    for result_table in csv_files:
        if result_table != (path + 'results.csv'):
            tables.append(pd.read_csv(result_table))
    pd.concat(tables, axis=0).to_csv(path + 'results.csv')


if __name__ == '__main__':
    concat_tables()

class T2V(Layer):
    """
    For each input feature, we apply the same layer in a time-independent (time-distributed layer) manner.
    This learnable embedding does not depend on time!
    Output dimension: 1 <= i <= k

    References:
    - https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    - https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/Time2Vec/Time2Vec.ipynb
    - https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
    """

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
        })
        return config
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                      shape=(input_shape[-1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.P = self.add_weight(name='P',
                      shape=(input_shape[1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.w = self.add_weight(name='w',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        self.p = self.add_weight(name='p',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)