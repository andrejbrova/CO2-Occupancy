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
        parameters,
        suffix=''
):
    result = {
        'batch size': parameters['batch_size'],
        'epochs': parameters['epochs'],
        'repeats': parameters['runs'],
        'embedding': parameters['embedding'],
        'feature set': parameters['feature_set'],
        'historical co2': parameters['historical_co2'],
        'accuracy_train_mean': np.mean(scores_train),
        'accuracy_train_std': np.std(scores_train),
        'accuracy_test_1_mean': np.mean([sublist[0] for sublist in scores_test_list]),
        'accuracy_test_1_std': np.std([sublist[0] for sublist in scores_test_list]),
    }

    if parameters['feature_set'] != '?' and parameters['feature_set'] != 'full':
        suffix += '_' + parameters['feature_set']
    if parameters['embedding'] != '?' and parameters['embedding'] != False:
        suffix += '_embedding'
    if parameters['dataset'] == '?' or parameters['dataset'] == 'uci':
        result.update({  # Add evaluation for second and combined test set
            'accuracy_test_2_mean': np.mean([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_2_std': np.std([sublist[1] for sublist in scores_test_list]),
            'accuracy_test_combined_mean': np.mean([sublist[2] for sublist in scores_test_list]),
            'accuracy_test_combined_std': np.std([sublist[2] for sublist in scores_test_list])
        })
        folder = 'results/'
    else:
        folder = 'results_' + parameters['dataset'] + '/'

    pd.DataFrame(result, index=[parameters['model_name']]).to_csv(ROOT_DIR + '/models/' + folder + parameters['model_name'] + suffix + '.csv')


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

class Time2Vec(Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],11),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],11),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[2], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[2], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))