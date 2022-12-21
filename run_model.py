from models import autoencoder
from models.embedding import embedding


def main():
    """
    Can run multiple models with predefined parameters.
    Model files can also be run individually.
    """

    # Include one or more parameter presets here
    parameter_presets = [
        #autoencoder_model_1_t2v,
        #autoencoder_model_2_t2v,
        #autoencoder_model_3_t2v,
        brick_australia_test_time2vec
    ]

    # Association of models to file, which can execute it
    model_files = {
        'autoencoder': autoencoder,
        'LSTM': embedding,
        'CNN': embedding
    }

    for parameters in parameter_presets:
        model = model_files[parameters['model_name']]
        model.run(parameters)


# Parameter presets are defined here:

autoencoder_model_1 = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'Light+CO2',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2 # 2 for plotting, 6 for brick dataset
}
autoencoder_model_2 = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': False,
    'windowing': False,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2
}
autoencoder_model_3 = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2
}
autoencoder_model_1_t2v = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'Light+CO2',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': True,
    'windowing': True,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2 # 2 for plotting, 6 for brick dataset
}
autoencoder_model_2_t2v = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': False,
    'windowing': True,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2
}
autoencoder_model_3_t2v = {
    'model_name': 'autoencoder',
    'dataset': 'uci',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 30,
    'historical_co2': False,
    'embedding': True,
    'windowing': True,
    'runs': 10,
    'batch_size': 16,
    'code_size': 2
}
brick_australia_CNN = {
    'model_name': 'CNN',
    'dataset': 'Australia',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_australia_LSTM = {
    'model_name': 'LSTM',
    'dataset': 'Australia',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_australia_autoencoder = {
    'model_name': 'autoencoder',
    'dataset': 'Australia',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
    'code_size': 6
}
brick_denmark_CNN = {
    'model_name': 'CNN',
    'dataset': 'Denmark',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_denmark_LSTM = {
    'model_name': 'LSTM',
    'dataset': 'Denmark',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_denmark_autoencoder = {
    'model_name': 'autoencoder',
    'dataset': 'Denmark',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
    'code_size': 6
}
brick_italy_CNN = {
    'model_name': 'CNN',
    'dataset': 'Italy',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_italy_LSTM = {
    'model_name': 'LSTM',
    'dataset': 'Italy',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
}
brick_italy_autoencoder = {
    'model_name': 'autoencoder',
    'dataset': 'Italy',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 200,
    'historical_co2': False,
    'embedding': True,
    'windowing': False,
    'runs': 10,
    'batch_size': 32,
    'code_size': 6
}
brick_australia_test_time2vec = {
    'model_name': 'CNN',
    'dataset': 'Australia',
    'feature_set': 'full',
    'split_data': True,
    'epochs': 1,
    'historical_co2': False,
    'embedding': True,
    'windowing': True,
    'runs': 1,
    'batch_size': 32,
}

if __name__ == '__main__':
    main()