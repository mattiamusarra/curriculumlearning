import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from config import CONFIG
from plotting_manager import PlottingManager

"""
CALLBACKS PERSONALIZZATE PER IL TRAINING CON CURRICULUM LEARNING E PLOTTING INTEGRATO - VERSIONE CORRETTA
"""

class RewardLoggerCallback(BaseCallback):
    """
    LOG DEI REWARD MEDI DURANTE IL TRAINING CON PLOTTING INTEGRATO - VERSIONE CORRETTA
    """
    def __init__(self, log_freq=None, verbose=1, plotting_manager=None):
        super().__init__(verbose)
        self.log_freq = log_freq or CONFIG.log_freq
        self.plotting_manager = plotting_manager
        self.reward_buffer = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Ottieni i reward originali dal buffer
            original_rewards = []
            curriculum_rewards = []
            
            if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
                original_rewards = self.locals['rewards']
            
            # METODO CORRETTO per ottenere i reward del curriculum
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    # Naviga attraverso i wrapper per trovare il CurriculumWrapper
                    current_env = env
                    curriculum_wrapper = None
                    
                    # Cerca il wrapper del curriculum nella catena di wrapper
                    while hasattr(current_env, 'env'):
                        if hasattr(current_env, '_last_curriculum_reward') and current_env._last_curriculum_reward is not None:
                            curriculum_wrapper = current_env
                            break
                        # Prova anche con il metodo get_wrapper_attr se disponibile
                        if hasattr(current_env, 'get_wrapper_attr'):
                            try:
                                last_curriculum_reward = current_env.get_wrapper_attr('_last_curriculum_reward')
                                if last_curriculum_reward is not None:
                                    curriculum_rewards.append(last_curriculum_reward)
                                    break
                            except:
                                pass
                        current_env = current_env.env
                    
                    # Se abbiamo trovato il wrapper, prendiamo il reward
                    if curriculum_wrapper:
                        curriculum_rewards.append(curriculum_wrapper._last_curriculum_reward)
                    
                    # Fallback: prova a ottenere dall'ultimo info se disponibile
                    if not curriculum_rewards and hasattr(current_env, 'last_info'):
                        if 'curriculum_reward' in current_env.last_info:
                            curriculum_rewards.append(current_env.last_info['curriculum_reward'])
            
            # Logging e salvataggio dati
            if curriculum_rewards:
                mean_curriculum_reward = np.mean(curriculum_rewards)
                mean_original_reward = np.mean(original_rewards) if original_rewards else 0
                
                print(f"Step {self.num_timesteps}: reward curriculum = {mean_curriculum_reward:.2f} (originale = {mean_original_reward:.2f})")
                
                # Salva nel plotting manager il reward del curriculum
                if self.plotting_manager:
                    self.plotting_manager.log_reward_data(self.num_timesteps, mean_curriculum_reward)
                    
            elif original_rewards:
                mean_original_reward = np.mean(original_rewards)
                print(f"Step {self.num_timesteps}: reward medio rollout = {mean_original_reward:.2f}")
                
                # Salva reward originale se non disponibile curriculum
                if self.plotting_manager:
                    self.plotting_manager.log_reward_data(self.num_timesteps, mean_original_reward)
            
            # RACCOLTA DATI GIUNTI - METODO MIGLIORATO
            if self.plotting_manager:
                self._collect_joint_data()
        
        return True
    
    def _collect_joint_data(self):
        """
        RACCOLTA DATI GIUNTI CON METODI MULTIPLI DI FALLBACK
        """
        joint_data_collected = False
        
        # Metodo 1: Prova con get_attr dal VecEnv
        if hasattr(self.training_env, 'get_attr') and not joint_data_collected:
            try:
                # Ottieni l'ambiente MuJoCo unwrapped
                unwrapped_envs = self.training_env.get_attr('unwrapped')
                if unwrapped_envs and hasattr(unwrapped_envs[0], 'data'):
                    env_data = unwrapped_envs[0].data
                    if hasattr(env_data, 'qpos') and hasattr(env_data, 'qvel'):
                        joint_pos = np.array(env_data.qpos.copy())
                        joint_vel = np.array(env_data.qvel.copy())
                        
                        self.plotting_manager.log_joint_data(
                            self.num_timesteps, joint_pos, joint_vel
                        )
                        joint_data_collected = True
                        
            except Exception as e:
                if self.verbose > 1:
                    print(f"Metodo 1 raccolta giunti fallito: {e}")
        
        # Metodo 2: Prova attraverso i wrapper dell'ambiente
        if not joint_data_collected and hasattr(self.training_env, 'envs'):
            try:
                env = self.training_env.envs[0]
                current_env = env
                
                # Naviga fino all'ambiente base MuJoCo
                while hasattr(current_env, 'env'):
                    if hasattr(current_env, 'data') or hasattr(current_env, 'sim'):
                        break
                    current_env = current_env.env
                
                # Prova a ottenere i dati dai diversi tipi di ambiente MuJoCo
                if hasattr(current_env, 'data'):
                    env_data = current_env.data
                elif hasattr(current_env, 'sim'):
                    env_data = current_env.sim.data
                elif hasattr(current_env, 'unwrapped') and hasattr(current_env.unwrapped, 'data'):
                    env_data = current_env.unwrapped.data
                else:
                    env_data = None
                
                if env_data and hasattr(env_data, 'qpos') and hasattr(env_data, 'qvel'):
                    joint_pos = np.array(env_data.qpos.copy())
                    joint_vel = np.array(env_data.qvel.copy())
                    
                    self.plotting_manager.log_joint_data(
                        self.num_timesteps, joint_pos, joint_vel
                    )
                    joint_data_collected = True
                    
            except Exception as e:
                if self.verbose > 1:
                    print(f"Metodo 2 raccolta giunti fallito: {e}")
        
        # Metodo 3: Fallback tramite osservazioni se disponibili
        if not joint_data_collected and hasattr(self.locals, 'obs_tensor'):
            try:
                # Le osservazioni potrebbero contenere informazioni sui giunti
                obs = self.locals['obs_tensor']
                if obs is not None and len(obs.shape) > 1 and obs.shape[1] >= 20:
                    # Estrai posizioni e velocità dalla struttura delle osservazioni
                    # Tipicamente le prime N/2 componenti sono posizioni, le successive velocità
                    n_joints = min(17, obs.shape[1] // 2)  # HumanoidStandup ha ~17 giunti
                    
                    joint_pos = obs[0, :n_joints].cpu().numpy() if hasattr(obs, 'cpu') else obs[0, :n_joints]
                    joint_vel = obs[0, n_joints:2*n_joints].cpu().numpy() if hasattr(obs, 'cpu') else obs[0, n_joints:2*n_joints]
                    
                    self.plotting_manager.log_joint_data(
                        self.num_timesteps, joint_pos, joint_vel
                    )
                    joint_data_collected = True
                    
            except Exception as e:
                if self.verbose > 1:
                    print(f"Metodo 3 raccolta giunti fallito: {e}")
        
        # Se tutti i metodi falliscono, log solo in verbose mode
        if not joint_data_collected and self.verbose > 1:
            print(f"Warning: Impossibile raccogliere dati giunti al step {self.num_timesteps}")


class CheckpointCallback(BaseCallback):
    """
    SALVATAGGIO PERIODICO DEI CHECKPOINT CON GENERAZIONE GRAFICI
    """
    
    def __init__(self, save_freq=None, save_path=None, plotting_manager=None, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq or CONFIG.checkpoint_freq
        self.save_path = save_path or CONFIG.model_folder
        self.plotting_manager = plotting_manager

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Salvataggio checkpoint originale
            checkpoint_file = os.path.join(self.save_path, "checkpoint_latest.zip")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f" Checkpoint salvato: {checkpoint_file} (step {self.num_timesteps})")
            
            # Genera grafico riassuntivo ad ogni checkpoint
            if self.plotting_manager:
                try:
                    print("Generazione grafico riassuntivo di training...")
                    self.plotting_manager.create_combined_summary_plot(save=True, show=False)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Errore generazione grafico riassuntivo: {e}")
                    pass
        
        return True


class PlottingCallback(BaseCallback):
    """
    CALLBACK DEDICATA ESCLUSIVAMENTE ALLA GENERAZIONE GRAFICI COMPLETI
    """
    
    def __init__(self, plotting_freq=None, plotting_manager=None, verbose=1):
        super().__init__(verbose)
        self.plotting_freq = plotting_freq or (CONFIG.checkpoint_freq * 2)  # Meno frequente dei checkpoint
        self.plotting_manager = plotting_manager

    def _on_step(self) -> bool:
        if self.plotting_manager and self.n_calls % self.plotting_freq == 0:
            try:
                if self.verbose > 0:
                    print(f"Generazione grafici completi (step {self.num_timesteps})...")
                
                # Genera tutti i grafici disponibili
                self.plotting_manager.generate_all_plots(show=False)
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Errore nella generazione grafici completi: {e}")
                pass
        
        return True


class DetailedCurriculumMonitorCallback(BaseCallback):
    """
    MONITORA IL PROGRESSO DETTAGLIATO DEL CURRICULUM LEARNING CON PLOTTING - VERSIONE CORRETTA
    """
    def __init__(self, check_freq=5000, verbose=1, plot_freq=50000, plotting_manager=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.plot_freq = plot_freq
        self.plotting_manager = plotting_manager
        
        #TRACCIAMENTO
        self.task_history = []
        self.height_history = []
        self.success_rates = []
        self.episode_lengths = []
        self.last_task = -1
        self.last_plot_step = 0
        
        #METRICHE PER TASK
        self.task_stats = {}
        
        print(" Monitor Curriculum con Plotting inizializzato")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._log_curriculum_progress()
            
        if self.n_calls % self.plot_freq == 0:
            self._create_progress_plots()
            # Genera anche i grafici del plotting manager
            if self.plotting_manager:
                self.plotting_manager.generate_all_plots()
            
        return True
    
    def _log_curriculum_progress(self):
        """LOG DEL PROGRESSO DEL CURRICULUM - VERSIONE CORRETTA"""
        if not hasattr(self.training_env, 'envs') or len(self.training_env.envs) == 0:
            return
            
        env = self.training_env.envs[0]
        
        #TROVA IL CURRICULUM WRAPPER - METODO CORRETTO
        current_env = env
        curriculum_wrapper = None
        
        # Naviga attraverso tutti i wrapper per trovare quello del curriculum
        while hasattr(current_env, 'env'):
            # Controlla se è il wrapper del curriculum
            if hasattr(current_env, 'get_detailed_curriculum_info'):
                curriculum_wrapper = current_env
                break
            # Controlla anche per gli attributi del curriculum base
            elif hasattr(current_env, 'current_task') and hasattr(current_env, 'tasks'):
                curriculum_wrapper = current_env
                break
            current_env = current_env.env
        
        # Controlla anche l'ultimo livello se non trovato
        if not curriculum_wrapper and hasattr(current_env, 'current_task'):
            curriculum_wrapper = current_env
        
        if not curriculum_wrapper:
            if self.verbose > 1:
                print(f"Warning: Curriculum wrapper non trovato al step {self.num_timesteps}")
            return
        
        # Ottieni info dettagliate se disponibili
        if hasattr(curriculum_wrapper, 'get_detailed_curriculum_info'):
            info = curriculum_wrapper.get_detailed_curriculum_info()
            self._log_detailed_info(info)
        else:
            # Fallback al sistema base
            self._log_basic_curriculum_info(curriculum_wrapper)
    
    def _log_detailed_info(self, info):
        """Log delle informazioni dettagliate del curriculum"""
        # Log del progresso
        print(f"\n CURRICULUM STATUS DETTAGLIATO (Step {self.num_timesteps:,}):")
        print(f"    Task {info['current_task']}/{info['total_tasks']}: {info['task_name']}")
        print(f"    Progresso: {info['success_count']}/{info['success_target']} successi")
        print(f"    Success rate: {info['success_rate']:.1%}")
        print(f"    Episodi: {info['episode_count']}")
        print(f"    Range episodi: {info['min_episodes']}-{info['max_episodes']}")
        print(f"    Altezza max: {info['best_height']:.2f}")
        
        if info['current_task'] == 8:  #TASK FINALE
            print(f"    Max equilibrio: {info['max_steps_standing']} step (target: {info['current_threshold']})")
        else:
            print(f"    Soglia: {info['current_threshold']}")
        
        #TRACCIA IL CAMBIO DI TASK
        current_task = info['current_task']
        if current_task != self.last_task:
            self.task_history.append((self.num_timesteps, current_task, info['task_name']))
            if current_task > self.last_task:
                print(f"\n TASK ADVANCEMENT RILEVATO!")
                print(f"   Nuovo Task {current_task}: {info['task_name']}")
            self.last_task = current_task
        
        #SALVA LE STATISTICHE DEL TASK
        self.task_stats[self.num_timesteps] = info.copy()
        
        #TRACCIA METRICHE PER I GRAFICI
        self.height_history.append((self.num_timesteps, info['best_height']))
        self.success_rates.append((self.num_timesteps, info['success_rate']))
        
        #AVVISI SPECIALI
        if info['episode_count'] > info['max_episodes'] * 0.8:
            remaining = info['max_episodes'] - info['episode_count']
            print(f" ATTENZIONE: {remaining} episodi rimanenti per timeout")
            
        if info['success_rate'] < 0.05 and info['episode_count'] > 200:
            print(f" SUCCESS RATE MOLTO BASSA - Possibile problema nel task")
    
    def _log_basic_curriculum_info(self, curriculum_wrapper):
        """Fallback per curriculum wrapper base"""
        current_task = curriculum_wrapper.current_task
        if hasattr(curriculum_wrapper, 'tasks') and current_task < len(curriculum_wrapper.tasks):
            task_name = curriculum_wrapper.tasks[current_task]['name']
            success_count = getattr(curriculum_wrapper, 'task_success_count', 0)
            success_target = curriculum_wrapper.tasks[current_task]['success_target']
            
            print(f"\n CURRICULUM STATUS BASIC (Step {self.num_timesteps}):")
            print(f"   Task {current_task + 1}/8: {task_name}")
            print(f"   Successi: {success_count}/{success_target}")
            
            if hasattr(curriculum_wrapper, 'max_steps_standing'):
                print(f"   Max steps in piedi: {curriculum_wrapper.max_steps_standing}")
            
            # Traccia cambio task
            if current_task != self.last_task:
                self.task_history.append((self.num_timesteps, current_task + 1, task_name))
                if current_task > self.last_task:
                    print(f"\n NUOVA TASK SBLOCCATA: Task {current_task + 1} - {task_name}!")
                self.last_task = current_task
    
    def _create_progress_plots(self):
        """CREA GRAFICI DEL PROGRESSO DEL CURRICULUM"""
        if len(self.task_stats) < 2:
            return
            
        try:
            #CREA CARTELLA PER I GRAFICI
            plots_dir = os.path.join("logs", "curriculum_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Curriculum Progress - Step {self.num_timesteps:,}', fontsize=16)
            
            #1. TASK PROGRESSION
            if self.task_history:
                steps, tasks, names = zip(*self.task_history)
                ax1.step(steps, tasks, where='post', linewidth=2, marker='o')
                ax1.set_xlabel('Training Steps')
                ax1.set_ylabel('Current Task')
                ax1.set_title('Task Progression')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 9)
            
            #2. HEIGHT PROGRESSION
            if self.height_history:
                steps, heights = zip(*self.height_history)
                ax2.plot(steps, heights, linewidth=2, color='green')
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Max Height Achieved')
                ax2.set_title('Height Progress')
                ax2.grid(True, alpha=0.3)
                
                #AGGIUNGI LINEE DI RIFERIMENTO 
                ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Squat height')
                ax2.axhline(y=1.17, color='red', linestyle='--', alpha=0.7, label='Standing height')
                ax2.legend()
            
            #3. SUCCESS RATE
            if self.success_rates:
                steps, rates = zip(*self.success_rates)
                ax3.plot(steps, rates, linewidth=2, color='blue')
                ax3.set_xlabel('Training Steps')
                ax3.set_ylabel('Success Rate')
                ax3.set_title('Task Success Rate')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 1)
            
            #4. TASK DISTRIBUTION
            task_counts = {}
            for step, info in self.task_stats.items():
                task = info['current_task']
                task_counts[task] = task_counts.get(task, 0) + 1
            
            if task_counts:
                tasks = list(task_counts.keys())
                counts = list(task_counts.values())
                ax4.bar(tasks, counts, color='purple', alpha=0.7)
                ax4.set_xlabel('Task Number')
                ax4.set_ylabel('Time Spent (checks)')
                ax4.set_title('Time Distribution Across Tasks')
                ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            #SALVA IL GRAFICO
            plot_path = os.path.join(plots_dir, f'curriculum_progress_step_{self.num_timesteps}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose > 0:
                print(f"Grafico progresso salvato: {plot_path}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Errore nella creazione grafici: {e}")
    
    def _on_training_end(self):
        """CREA REPORT FINALE DEL CURRICULUM E GRAFICI FINALI"""
        try:
            # Report finale curriculum
            print("\n" + "="*60)
            print("REPORT FINALE CURRICULUM LEARNING")
            print("="*60)
            
            if not self.task_stats:
                print("Nessun dato raccolto")
                return
            
            #ANALISI FINALE
            final_info = list(self.task_stats.values())[-1]
            
            print(f"Task finale raggiunto: {final_info['current_task']}/8")
            print(f"Nome task: {final_info['task_name']}")
            print(f"Altezza massima: {final_info['best_height']:.2f}")
            if 'max_steps_standing' in final_info:
                print(f"Steps massimi in equilibrio: {final_info['max_steps_standing']}")
            
            #TEMPO SPESO PER LE TASK
            print(f"\nDistribuzione tempo per task:")
            task_time = {}
            for info in self.task_stats.values():
                task = info['current_task']
                task_time[task] = task_time.get(task, 0) + 1
            
            for task, time in sorted(task_time.items()):
                percentage = (time / len(self.task_stats)) * 100
                print(f"   Task {task}: {percentage:.1f}% del tempo")
            
            #GENERA GRAFICI FINALI
            if self.plotting_manager:
                print("\nGenerazione grafici finali...")
                final_plots = self.plotting_manager.generate_all_plots()
                print(f"Generati {len(final_plots)} grafici finali")
            
            #SALVA IL REPORT FINALE
            report_path = os.path.join("logs", "curriculum_final_report.txt")
            os.makedirs("logs", exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write("CURRICULUM LEARNING FINAL REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Final Task: {final_info['current_task']}/8\n")
                f.write(f"Task Name: {final_info['task_name']}\n")
                f.write(f"Max Height: {final_info['best_height']:.2f}\n")
                if 'max_steps_standing' in final_info:
                    f.write(f"Max Standing Steps: {final_info['max_steps_standing']}\n")
                f.write(f"Total Training Steps: {self.num_timesteps:,}\n\n")
                
                f.write("TIME DISTRIBUTION:\n")
                for task, time in sorted(task_time.items()):
                    percentage = (time / len(self.task_stats)) * 100
                    f.write(f"  Task {task}: {percentage:.1f}%\n")
                
                f.write(f"\nTASK TRANSITIONS:\n")
                for step, task, name in self.task_history:
                    f.write(f"  Step {step:,}: Advanced to Task {task} - {name}\n")
            
            print(f"Report finale salvato: {report_path}")
            
        except Exception as e:
            print(f"Errore nel report finale: {e}")


class ImprovedCallbackManager:
    """GESTISCE TUTTE LE CALLBACKS PER IL TRAINING CON CURRICULUM E PLOTTING ROBUSTO - VERSIONE CORRETTA"""
    
    def __init__(self, eval_env, config=CONFIG, enable_plotting=True):
        self.config = config
        self.eval_env = eval_env
        self.enable_plotting = enable_plotting
        
        # Inizializza il plotting manager se richiesto
        self.plotting_manager = None
        if self.enable_plotting:
            try:
                self.plotting_manager = PlottingManager(config)
                print("PlottingManager con supporto curriculum inizializzato correttamente")
            except Exception as e:
                print(f"Warning: Impossibile inizializzare PlottingManager: {e}")
                print("Continuando senza generazione grafici...")
                self.enable_plotting = False
        
    def create_callbacks(self):
        """
        CREA E RESTITUISCE TUTTE LE CALLBACK CON PLOTTING E CURRICULUM INTEGRATI
        """
        os.makedirs(self.config.model_folder, exist_ok=True)
        os.makedirs(self.config.best_model_folder, exist_ok=True)
        os.makedirs(self.config.logs_folder, exist_ok=True)
        
        # Callback base con supporto plotting robusto
        reward_logger = RewardLoggerCallback(
            plotting_manager=self.plotting_manager if self.enable_plotting else None,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            plotting_manager=self.plotting_manager if self.enable_plotting else None
        )
        
        # Monitor curriculum dettagliato con plotting integrato
        curriculum_monitor = DetailedCurriculumMonitorCallback(
            check_freq=5000, 
            plot_freq=25000,
            plotting_manager=self.plotting_manager if self.enable_plotting else None,
            verbose=1
        )

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.config.best_model_folder,
            log_path=self.config.logs_folder,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        
        callbacks_list = [reward_logger, eval_callback, checkpoint_callback, curriculum_monitor]
        
        # Aggiungi callback dedicata ai grafici se abilitata
        if self.enable_plotting and self.plotting_manager:
            plotting_callback = PlottingCallback(
                plotting_manager=self.plotting_manager
            )
            callbacks_list.append(plotting_callback)
        
        print(f"Callback manager inizializzato con {len(callbacks_list)} callbacks")
        if self.enable_plotting:
            print("- Plotting integrato abilitato")
        print("- Curriculum monitoring abilitato")
        
        return callbacks_list
    
    def get_plotting_manager(self):
        """Restituisce il plotting manager per uso esterno"""
        return self.plotting_manager
    
    def finalize_plotting(self):
        """
        GENERA I GRAFICI FINALI AL TERMINE DEL TRAINING
        """
        if self.enable_plotting and self.plotting_manager:
            try:
                print("\n" + "="*60)
                print("GENERAZIONE GRAFICI FINALI CON DATI CURRICULUM")
                print("="*60)
                
                generated_plots = self.plotting_manager.generate_all_plots(show=False)
                
                if generated_plots:
                    print(f"Grafici finali generati con successo!")
                    print(f"Totale grafici: {len(generated_plots)}")
                    for plot_path in generated_plots:
                        print(f"  - {plot_path}")
                else:
                    print("Nessun grafico generato - possibili problemi con i dati")
                
                return generated_plots
                
            except Exception as e:
                print(f"Errore nella generazione grafici finali: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            print("Plotting non abilitato - nessun grafico generato")
            return []