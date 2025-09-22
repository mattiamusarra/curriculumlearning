"""
IN QUESTA CLASSE ANDIAMO A GESTIRE GLI AMBIENTI DI TRAINING E VALUTAZIONE
"""
import gymnasium as gym 
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from config import CONFIG
from utils import get_env_seed
from curriculum_wrapper import CurriculumWrapper
import os
import glfw
from datetime import datetime
import shutil


class Float32Wrapper(gym.ObservationWrapper):
    """
    USIAMO UN WRAPPER PER FORZARE LE OSSERVAZIONI A FLOAT32, PERCHé LE GPU APPLE NON SUPPORTANO FLOAT64
    INOLTRE ANCHE SE USIAMO LA CPU OCCUPA MENO RAM E PERMETTE QUINDI DI ESEGUIRE PIÙ OPERAZIONI.
    TALE CLASSE ESTENDE ObservationWrapper di Gym.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.astype(np.float32),
            high=self.observation_space.high.astype(np.float32),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        LA FUNZIONE VIENE USATA PER CONVERTIRE TUTTE LE OSSERVAZIONI DELL'AMBIENTE IN FORMATO FLOAT32".
        """
        return obs.astype(np.float32)

class EnvironmentManager:
    """
    MEDIANTE QUESTA CLASSE ANDIAMO A GESTIRE E CONFIGURARE GLI AMBIENTI
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.glfw_initialized = False
        self.initialize_glfw()

    def initialize_glfw(self):
        """
        INIZIALIZZA GLFW PER LA GESTIONE DI ERRORI DI RENDERING
        """
        try:
            if not glfw.init():
                print("Warning: Impossibile inizializzare GLFW")
                self.glfw_initialized = False
            else:
                #IMPOSTIAMO IL CONTESTO PER IL RENDERING HEADLESS
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                self.glfw_initialized = True
                print("GLFW inizializzato correttamente")
        except Exception as e:
            print(f"Warning: Errore nell'inizializzazione di GLFW: {e}")
            self.glfw_initialized = False

    def _clean_video_folder(self, folder_path):
        """
        PULISCE LA CARTELLA VIDEO PER EVITARE WARNING DI SOVRASCRITTURA
        """
        if os.path.exists(folder_path):
            #CREA BACKUP DELLA CARTELLA ESISTENTE
            backup_folder = f"{folder_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.listdir(folder_path):  # SE LA CARTELLA NON E' VUOTA
                print(f"Backup video esistenti in: {backup_folder}")
                shutil.copytree(folder_path, backup_folder)
            
            #RIMOZIONE CARTELLA ORIGINALE
            shutil.rmtree(folder_path)
        
        #RICREA LA CARTELLA 
        os.makedirs(folder_path, exist_ok=True)

    def create_single_env(self, render_mode="rgb_array", seed=None, use_curriculum=False):
        """
        ANDIAMO A CREARE UN AMBIENTE SINGOLO
        """
        try:
            #PROVIAMO PRIMA CON IL RENDER MODE RICHIESTO
            env = gym.make(self.config.env_id, render_mode=render_mode)
            
            #IMPOSTA IL SEED SE FORNITO
            if seed is not None:
                env.reset(seed=seed)
                env.action_space.seed(seed)
            
            #APPLICA IL CURRICULUM WRAPPER SE RICHIESTO
            if use_curriculum:
                env = CurriculumWrapper(env)
                print("Curriculum learning attivato!")
                
            env = Monitor(env)
            wrapped_env = Float32Wrapper(env)
            
            #TEST DEL RENDERING PER VEDERE SE FUNZIONA
            if render_mode == "rgb_array":
                try:
                    obs, _ = wrapped_env.reset()
                    test_frame = wrapped_env.render()
                    if test_frame is None:
                        print("Warning: Il rendering restituisce None, provo fallback")
                        wrapped_env.close()
                        raise Exception("Rendering fallito")
                except Exception as render_test_error:
                    print(f"Test rendering fallito: {render_test_error}")
                    wrapped_env.close()
                    raise render_test_error
            
            return wrapped_env
            
        except Exception as e:
            print(f"Errore nella creazione dell'ambiente con render_mode={render_mode}: {e}")
            
            #FALLBACK: proviamo SENZA RENDERING
            if render_mode != None:
                try:
                    print("Tentativo fallback senza rendering...")
                    env = gym.make(self.config.env_id, render_mode=None)
                    if seed is not None:
                        env.reset(seed=seed)
                        env.action_space.seed(seed)
                    
                    #APPLICA IL CURRRICULUM WRAPPER SE RICHIESTO
                    if use_curriculum:
                        env = CurriculumWrapper(env)
                        
                    env = Monitor(env)
                    return Float32Wrapper(env)
                except Exception as e2:
                    print(f"Errore anche nel fallback: {e2}")
                    raise
            else:
                raise

    def create_training_env(self):
        """CREA UN AMBIENTE DI TRAINING VETTORIALIZZATO CON CURRICULUM"""
        #CREA UNA LISTA DI FUNZIONI LAMBDA OGNUNA CON IL PROPRIO SEED E CURRICULM
        env_fns = []
        for i in range(self.config.n_envs):
            env_seed = get_env_seed(self.config.seed, i)
            # APPLICA IL CURRICULUM LEARNING
            env_fns.append(lambda seed=env_seed: self.create_single_env(seed=seed, use_curriculum=True))
        
        training_env = DummyVecEnv(env_fns)
        return training_env
    
    def create_video_env(self, base_env):
            """
            CREA UN AMBIENTE CON REGISTRAZIONE VIDEO
            """
            print(f"\n  CONFIGURAZIONE VIDEO ENVIRONMENT")
            print(f"   Cartella video: {self.config.video_folder}")
            print(f"   Intervallo: {self.config.video_interval}")
            print(f"   Lunghezza: {self.config.video_length}")
            
            #PULIAMO LA CARTELLA
            self._clean_video_folder(self.config.video_folder)
            
            #VERIFICA CHE LA CARTELLA ESISTA
            if not os.path.exists(self.config.video_folder):
                print(f" Errore: cartella {self.config.video_folder} non esiste!")
                return base_env
                
            print(f"Cartella video creata: {os.path.abspath(self.config.video_folder)}")
            
            try:
                def debug_trigger(step):
                    should_record = step > 0 and step % self.config.video_interval == 0
                    if should_record:
                        print(f" TRIGGER VIDEO ATTIVATO al step {step}!")
                    return should_record
                
                print("Creazione VecVideoRecorder...")
                
                vec_env = VecVideoRecorder(
                    base_env,
                    video_folder=self.config.video_folder,
                    record_video_trigger=debug_trigger,
                    video_length=self.config.video_length,
                    name_prefix="ppo_humanoid"
                )
                
                print(" VecVideoRecorder creato con successo!")
                
                #TEST IMMEDIATO
                print(" Test immediato del video recorder...")
                
                return vec_env
                
            except Exception as e:
                print(f" ERRORE nella creazione video environment: {e}")
                print(f"   Tipo errore: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("  Continuando senza video...")
                return base_env
        
    def create_eval_env(self):
        """
        CREA L'AMBIENTE DI VALUTAZIONE
        """
        #USA UN SEED DIVERSO PER LA VALUTAZIONE
        eval_seed = get_env_seed(self.config.seed, 1000) if self.config.seed else None
        return DummyVecEnv([lambda: self.create_single_env(seed=eval_seed)])
    
    def create_test_env(self):
        """
        CREA L'AMBIENTE DI TEST PER I VIDEO DI SUCCESSO )
        """
        try:
            print("Creando ambiente di test semplificato...")
            
            #PULIAMO LA CARTELLA
            self._clean_video_folder(self.config.success_video_folder)
            
            #USA UN SEED DIVERSO PER IL TEST
            test_seed = get_env_seed(self.config.seed, 2000) if self.config.seed else None
            env = self.create_single_env(render_mode="rgb_array", seed=test_seed)
            
            print("Ambiente di test creato con successo")
            return env
            
        except Exception as e:
            print(f"Warning: Errore nella creazione ambiente test: {e}")
            print("Tentativo con ambiente senza rendering...")
            
            #FALLBACK: AMBIENTE SENZA RENDERING
            try:
                test_seed = get_env_seed(self.config.seed, 2000) if self.config.seed else None
                return self.create_single_env(render_mode=None, seed=test_seed)
            except Exception as e2:
                print(f"Errore anche nel fallback: {e2}")
                raise

    def __del__(self):
        """
        CLEANUP QUANDO L'OGGETTO VIENE DISTRUTTO
        """
        try:
            if self.glfw_initialized:
                glfw.terminate()
                print("GLFW terminato correttamente")
        except Exception as e:
            print(f"Warning durante cleanup GLFW: {e}")
            pass