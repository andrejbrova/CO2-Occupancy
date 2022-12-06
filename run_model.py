from models import autoencoder


def main():
    """
    Can run multiple models with predefined parameters
    """

    # Include one or more parameter presets here
    parameter_presets = [
        autoencoder_model_1,
        autoencoder_model_2,
        autoencoder_model_3,
    ]

    for parameters in parameter_presets:
        autoencoder.run(parameters)


# Parameter presets are defined here:

autoencoder_model_1 = {
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
autoencoder_model_2 = {
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
autoencoder_model_3 = {
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