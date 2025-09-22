"""
UTILITÀ PER LA GESTIONE DEI SEED E RIPRODUCIBILITÀ
"""
import random
import numpy as np
import torch
import os
from typing import Optional


def set_all_seeds(seed: Optional[int]):
    """
    IMPOSTA TUTTI I SEED PER GARANTIRE LA RIPRODUCIBILITÀ
    """
    if seed is None:
        print("Seed non impostato - esperimenti non riproducibili")
        return
    
    print(f"Impostazione seed globale: {seed}")
    
    #IMPOSTA IL SEED PER IL MODULO STANDARD RANDOM DI PYTHON
    random.seed(seed)
    
    #IMPOSTA IL SEED PER IL GENERATORE CASUALE DI NUMPY
    np.random.seed(seed)
    
    #IMPOSTA IL SEED PER IL GENERATORE CASUALE DI PYTORCH
    torch.manual_seed(seed)
    
    #PER CUDA (INUTILIZZATO)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Per determinismo completo con CUDA (può ridurre le prestazioni)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    #SERVE PER RENDERE DETERMINISTICO L'HASHING DI STIRNGHE E ALTRE STRUTTURE CHE PYTHON ALTRIMENTI RANDOMIZZEREBBE
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("Tutti i seed sono stati impostati con successo")


def get_env_seed(base_seed: Optional[int], env_index: int) -> Optional[int]:
    """
    GENERA UN SEED UNICO PER OGNI AMBIENTE PARALLELO
    """
    if base_seed is None:
        return None
    return base_seed + env_index * 1000