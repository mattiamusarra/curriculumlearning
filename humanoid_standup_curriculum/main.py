from config import CONFIG
from environment import EnvironmentManager  
from model_manager import ModelManager
from callbacks import ImprovedCallbackManager
from success_video import SuccessVideoRecorder
from plotting_manager import PlottingManager
from utils import set_all_seeds
import os
import platform
from stable_baselines3 import PPO

#CONFIGURAZIONE HEADLESS - DEVE ESSERE FATTA PRIMA DI QUALSIASI IMPORT DI GYMNASIUM/MUJOCO
os.environ['PYOPENGL_PLATFORM'] = 'darwin'  #PER MACOS
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def test_rendering():
    """TEST SE IL RENDERING FUNZIONA"""
    print("\nTESTING RENDERING...")
    try:
        import gymnasium as gym
        
        #TEST AMBIENTE SEMPLICE
        env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
        obs, _ = env.reset()
        
        #PROVA DI RENDERING
        frame = env.render()
        
        if frame is None:
            print("Rendering restituisce None")
            return False
            
        if not hasattr(frame, 'shape'):
            print("Frame non ha shape")
            return False
            
        print(f"Rendering OK! Frame shape: {frame.shape}")
        env.close()
        return True
        
    except Exception as e:
        print(f"Errore rendering: {e}")
        return False

def main():
    #IMPOSTA I SEED PER LA RIPRODUCIBILITÀ
    set_all_seeds(CONFIG.seed)
    
    if not test_rendering():
        print("WARNING: Rendering non funziona - video training disabilitati")
    
    print(f"Sistema operativo: {platform.system()}")
    print(f"Dispositivo utilizzato: {CONFIG.torch_device}")
    if CONFIG.seed is not None:
        print(f"Seed utilizzato: {CONFIG.seed}")
    print("="*50)
    
    try:
        #INIZIALIZZA I MANAGER
        print("Inizializzazione manager...")
        env_manager = EnvironmentManager()
        model_manager = ModelManager()
        
        # INIZIALIZZA PLOTTING MANAGER
        plotting_manager = PlottingManager(CONFIG)
        print("PlottingManager inizializzato")
        
        video_recorder = SuccessVideoRecorder()
        
        #CREAZIONE DEGLI AMBIENTI
        print("Creazione ambienti...")
        train_env = env_manager.create_training_env()
        vec_env = env_manager.create_video_env(train_env)
        eval_env = env_manager.create_eval_env()
        
        #CREAZIONE DELLE CALLBACK CON PLOTTING INTEGRATO
        print("Creazione callback con plotting integrato...")
        callback_manager = ImprovedCallbackManager(eval_env)
        callbacks = callback_manager.create_callbacks()
        
        # OTTIENI IL PLOTTING MANAGER DAL CALLBACK MANAGER
        callback_plotting_manager = callback_manager.get_plotting_manager()
        
        #CAPIAMO SE ESISTE GIÀ UN MODELLO O DOBBIAMO CREARNE UNO
        print("Gestione modelli...")
        model, remaining_steps = model_manager.load_or_create_model(train_env)
        
        should_train = model_manager.should_continue_training(remaining_steps)
        
        if should_train:
            if remaining_steps == 0:
                remaining_steps = CONFIG.total_timesteps
                print(f"Training forzato per {remaining_steps:,} steps...")
            else:
                print(f"Inizio training per {remaining_steps:,} steps...")
            
            #INIZIA IL TRAINING
            print("\n TRAINING CON CURRICULUM LEARNING E PLOTTING - SOTTO-TASK GRADUALI")
            print("Task 1: Controllo del torso - Imparare a sollevare il busto")
            print("Task 2: Posizione accovacciata - Portare le gambe sotto il corpo")
            print("Task 3: Alzarsi in piedi - Usare gambe e braccia per alzarsi")
            print("Task 4: Rimanere in piedi - Mantenere l'equilibrio per 50+ step")
            print("I reward del curriculum e i dati dei giunti saranno registrati per i grafici\n")
            
            model.set_env(vec_env)
            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
            )

            #SALVATAGGIO MODELLO FINALE
            model_manager.save_final_model(model)
            
        else:
            print("Training saltato - modello già completato")
        
        #CHIUDIAMO GLI AMBIENTI DI TRAINING
        print("Chiusura ambienti di training...")
        try:
            vec_env.close()
            eval_env.close() 
            train_env.close()
        except Exception as e:
            print(f"Warning durante chiusura ambienti: {e}")
        
        #GENERAZIONE GRAFICI FINALI
        print("\n" + "="*50)
        print("GENERAZIONE GRAFICI FINALI")
        print("="*50)
        
        try:
            # Usa il metodo finalize_plotting del callback manager
            final_plots = callback_manager.finalize_plotting()
            
            if final_plots:
                print(f"Tutti i grafici sono stati salvati nelle rispettive cartelle")
            else:
                print("Nessun grafico finale generato - controlla i log per eventuali errori")
                
        except Exception as e:
            print(f"Errore durante generazione grafici finali: {e}")
            import traceback
            traceback.print_exc()
        
        #CREAZIONE VIDEO DI SUCCESSO
        print("\n" + "="*50)
        print("PREPARAZIONE CREAZIONE VIDEO")
        print("="*50)
        try:
            success = video_recorder.create_success_video(model, env_manager)
            if success:
                print("Video di successo creato!")
            else:
                print("Nessun video di successo creato")
        except Exception as e:
            print(f"Errore durante creazione video: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("ESECUZIONE COMPLETATA")
        print("="*60)
        print("Controlla le cartelle:")
        print("  - 'plots/' per i grafici generati")
        print("  - 'training_data/' per i dati grezzi")
        print("  - 'logs/' per i report del curriculum")
        print("  - 'videos/' per i video di successo")
        print("="*60)
        
    except Exception as main_error:
        print(f"Errore principale: {main_error}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            import glfw
            glfw.terminate()
        except:
            pass

if __name__ == "__main__":
    main()