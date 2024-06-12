from ..qata.random_split import split as split_qata
from ..cervix.random_split import split as split_cervix

def get_split_fn(data_name):
    if data_name in ['qata', 'polyp']:
        return split_qata

    elif data_name in ['cervix']:
        return lambda _, x: split_cervix(x)

    raise ValueError(f'Unknown data_name: {data_name}')