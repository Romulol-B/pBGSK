from ucimlrepo import fetch_ucirepo
from pandas import DataFrame

DATASET_REGISTRY = {
    "tic_tac_toe_endgame": 101,
    #"breast_cancer": 15,
    #"wine_quality": 109,
    "heart_disease": 45,
    "house_votes": 105,
    "zoo": 111,
    "lymphography": 63,
    "hepatitis": 46,
    # medium size datasets
    "waveform": 107,
    #"german_credit": 31,
    "wdbc": 857,
    "ionosphere": 52,
    "dermatology": 33,
    #"soybean": 91,
    #"lung_cancer": 62,
    #"spambase": 94,
    "sonar": 151,  # ,
    # large size datasets
    # "hill_valley":  106 , sem import direto
     "clean1":  189 ,# gallstone
     #"semeion":  49 ,
     #"arrhythmia":  148 ,
     #"cnae":  192
}


def data_loader(dataset_name: str) -> DataFrame:
    """ Importa um dos datasets utilizados pelo artigo para benchmark.
        
        Args:
            dataset_name: Nome do dataset no registro.
            
        Returns:
            O objeto dataset do ucimlrepo.
            
        Raises:
            ValueError: Se o dataset não for encontrado no registro.
     """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry.")
    
    return fetch_ucirepo(id=DATASET_REGISTRY[dataset_name])