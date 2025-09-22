"""
CONFIGURAZIONE PER IL TRAINING PPO
"""
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """
    PARAMETRI UTILIZZATI NEL CODICE
    """
    env_id : str = "HumanoidStandup-v5" #AMBIENTE SCELTO
    total_timesteps: int = 60_000_000 #NUMERO DI PASSI TOTALI PER IL TRAINING
    n_envs: int = 8 #NUMERO DI EPISODI ESEGUITI IN PARALLELO COSI' L'AGENTE PUÒ IMPARARE PIÙ RAPIDAMENTE
    device: str = "cpu" #UTILIZZIAMO LA CPU PER IL TRAINING
    
    #SEED PER RIPRODUCIBILITÀ 
    seed: Optional[int] = 42  #SE È NON NON RIPRODUCIBILITÀ ALTRIMENTI NUMERO

    #PERCORSI DI SALVATAGGIO VIDEO E MODELLI
    video_folder: str = "video_training"
    success_video_folder: str = "video_success"
    model_folder: str = "models"
    best_model_folder: str = "best_model"
    logs_folder: str = "logs"

    #PARAMETRI DI REGISTRAZIONE VIDEO
    video_interval: int = 100_000
    video_length: int = 1000

    #PARAMETRI DI VALUTAZIONE
    eval_freq: int = 50_000 #PERMETTE DI VALUTARE SULL'AMBIENTE DI TEST
    checkpoint_freq: int = 50_000 #PERMETTE DI SALVARE UN CHECKPOINT DEL MODELLO ATTUALE
    log_freq: int = 10_000 #PERMETTE DI STAMPARE INFORMAZIONI SUL REWARD MEDIO DEL TRAINING
    n_eval_episodes: int = 3 #Permette di valutare il modello su 5 episodi completi

    #SUCCESSO
    success_threshold: int = 70 #IL MANICHINO DOVRÀ RIMANERE IN PIEDI PER ALMENO TOT STEPS PER DICHIARARE UN SUCCESSO
    max_test_episodes: int = 2_000  #NUMERO MASSIMO DI TENTATIVI PER TROVARE UN EPISODIO CHE ABBIA SUCCESSO
    max_test_steps: int = 100 #NUMERO MASSIMO DI STEP PER OGNI SINGOLO EPISODIO DI TEST

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


CONFIG = TrainingConfig()