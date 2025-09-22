"""
GESTIONE DEI MODELLI : CARICAMENTO, SALVATAGGIO E INFORMAZIONI
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from config import CONFIG

class ModelManager:
    """
    GESTIONE DELLE OPERAZIONI
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.model_folder = config.model_folder
        self.best_model_folder = config.best_model_folder
        self._ensure_directories()
    
    def _ensure_directories(self):
        """
        CREA LE DIRECTORIES NECESSARIE
        """
        for folder in [self.model_folder, self.best_model_folder]:
            os.makedirs(folder, exist_ok=True)
    
    @property
    def model_paths(self) -> Dict[str, str]:
        """
        RESTITUISCE I PERCORSI DEI MODELLI
        """
        return {
            'best': os.path.join(self.best_model_folder, "best_model.zip"),
            'final': os.path.join(self.model_folder, "ppo_humanoid_finale.zip"),
            'checkpoint': os.path.join(self.model_folder, "checkpoint_latest.zip")
        }
    
    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """
        OTTIENE INFORMAZIONI SUL MODELLO
        """
        if not os.path.exists(model_path):
            return None
        
        try:
            device = self.config.torch_device
            temp_model = PPO.load(model_path, device=device)
            timesteps = temp_model.num_timesteps if hasattr(temp_model, 'num_timesteps') else 0 #
            file_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            return {
                'path': model_path,
                'timesteps': timesteps,
                'mtime': os.path.getmtime(model_path), #data di ultima modifica
                'file_time': file_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Errore nel leggere {model_path}: {e}")
            return None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        RACCOGLIE LE INFORMAZIONI DISPONIBILI
        """
        models_info = {}
        
        #È STATA DEFINITO UN ORDINE DI PROPRIETÀ : CHECKPOINT -> BEST -> FINAL
        for name in ['checkpoint', 'best', 'final']:
            path = self.model_paths[name]
            info = self.get_model_info(path)
            if info:
                models_info[name] = info
        
        return models_info
    
    def choose_best_model(self, models_info: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        SCEGLIE IL MODELLO MIGLIORE DA CARICARE
        """
        if not models_info:
            return None
        candidates = []
        for model_type in ['checkpoint', 'best', 'final']:
            if model_type in models_info:
                candidates.append(models_info[model_type])
            
        best_model = max(candidates, key=lambda x: x['mtime'])
        return best_model    
    
    def load_or_create_model(self, train_env) -> tuple[PPO, int]:
        """
        CARICA UN MODELLO ESISTENTE O NE CREA UNO NUOVO
        """
        models_info = self.get_available_models()
        
        if not models_info:
            print("Nessun modello trovato. Creo nuovo modello PPO")
            device = self.config.torch_device
            model = PPO(
                "MlpPolicy", 
                train_env, 
                verbose=1, 
                device=device,
                n_steps=4096,
                batch_size=128,
                learning_rate=3e-4,
                seed=self.config.seed  # IMPOSTA IL SEED PER IL MODELLO
            )
            total_timesteps = self.config.total_timesteps
            return model, total_timesteps
        
        #MOSTRA I MODELLI DISPONIBILI
        print("Modelli disponibili:")
        for name, info in models_info.items():
            print(f"  - {name}: {info['timesteps']:,} steps, modificato il {info['file_time']}")
        
        #SCEGLI IL MODELLO MIGLIORE
        chosen_model = self.choose_best_model(models_info)
        if not chosen_model:
            #FALLBACK : CREIAMO UN NUOVO MODELLO
            device = self.config.torch_device
            model = PPO(
                "MlpPolicy", 
                train_env, 
                verbose=1, 
                device=device,
                n_steps=4096,
                batch_size=128,
                learning_rate=5e-4,
                seed=self.config.seed  #IMPOSTA IL SEED PER IL MODELLO
            )
            total_timesteps = self.config.total_timesteps
            return model, total_timesteps
        
        print(f"Carico modello da {chosen_model['path']}")
        print(f"    Steps già completati: {chosen_model['timesteps']:,}")
        
        device = self.config.torch_device
        model = PPO.load(chosen_model['path'], env=train_env, device=device)
        
        #CALCOLA GLI STEP RIMANENTI
        total_timesteps = self.config.total_timesteps
        remaining_steps = max(0, total_timesteps - chosen_model['timesteps'])
        if remaining_steps > 0:
            print(f"Rimangono {remaining_steps:,} steps da completare")
        else:
            print("Training già completato!")
        
        return model, remaining_steps
    
    def save_final_model(self, model: PPO):
        """
        SALVA IL MODELLO FINALE
        """
        final_path = self.model_paths['final']
        model.save(final_path)
        print(f"Modello finale salvato in {final_path}")
        
    #ELIMINARE
    def should_continue_training(self, remaining_steps: int) -> bool:
        """Determina se continuare il training"""
        if remaining_steps > 0:
            return True
        
        force_training = input(
            "Vuoi forzare il training anche se sembra completato? (s/n): "
        ).lower().startswith('s')
        
        return force_training