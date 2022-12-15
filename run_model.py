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
        autoencoder_model_3_t2v,
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

if __name__ == '__main__':
    main()