
"""
Gestione della creazione di video di successo
"""
import os
import imageio
import shutil
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from config import CONFIG
import numpy as np 
from environment import EnvironmentManager
import csv
class SuccessVideoRecorder:
    """
    REGISTRA VIDEO CHE RISPECCHIANO IL NOSTRO CONCETTO DI SUCCESSO:
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self._setup_video_folder()

    def _setup_video_folder(self):
        """
        CONFIGURA LA CARTELLA VIDEO CON GESTIONE BACKUP
        """
        #PULISCI LA CARTELLA ESISTENTE
        if os.path.exists(self.config.success_video_folder):
            shutil.rmtree(self.config.success_video_folder)
        os.makedirs(self.config.success_video_folder, exist_ok=True)

    def _is_standing(self, obs) -> bool:
        """
        CONTROLLIAMO SE IL MANICHINO È IN PIEDI
        """
        z_position = obs[0] if len(obs) > 0 else 0
        return z_position > 1.2
    
    def _get_stability_info(self, obs) -> dict:
        """
        OTTIENE INFORMAZIONI SULLA STABILITÀ DEL MANICHINO
        """
        if len(obs) < 6:
            return {'z_pos': 0, 'stability': 0, 'is_stable': False}
        
        z_pos = obs[0]
        #VELOCITA ANGOLARE
        angular_vel = np.linalg.norm(obs[1:4]) if len(obs) > 3 else 0
        
        is_stable = z_pos > 1.2 and angular_vel < 2.0
        stability_score = max(0, z_pos - angular_vel * 0.1)
        
        return {
            'z_pos': z_pos,
            'angular_vel': angular_vel,
            'stability': stability_score,
            'is_stable': is_stable
        }
    


    def record_success_episode(self, model: PPO, test_env, episode: int) -> tuple[bool, int, int, list]:
        """
        REGISTRA UN EPISODIO, RACCOGLIE I FRAME E SALVA I SEGNALI DEI GIUNTI
        """
        frames = []
        torques_log_path = f"joint_torques_episode_{episode}.csv"

        try:
            obs, _ = test_env.reset()
            done, truncated = False, False
            standing_counter = 0
            max_standing = 0
            step_count = 0

            print(f"  Episodio {episode}: inizio registrazione...")

            with open(torques_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Torques", "Action", "qpos", "qvel"])

                while not (done or truncated) and step_count < self.config.max_test_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = test_env.step(action)
                    step_count += 1

                    torques = test_env.unwrapped.data.actuator_force.copy().tolist()
                    qpos = test_env.unwrapped.data.qpos.copy().tolist()
                    qvel = test_env.unwrapped.data.qvel.copy().tolist()
                    writer.writerow([step_count, torques, action.tolist(), qpos, qvel])

                    #REGISTRA FRAME OGNI POCHI STEP PER RIDURRE MEMORIA
                    if step_count % 2 == 0:
                        try:
                            frame = test_env.render()
                            if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                                frames.append(frame)
                        except Exception as render_error:
                            print(f"Warning: Errore nel rendering al step {step_count}: {render_error}")
                            continue

                    stability = self._get_stability_info(obs)

                    if stability['is_stable']:
                        standing_counter += 1
                        max_standing = max(max_standing, standing_counter)
                    else:
                        if standing_counter > 0:
                            print(f"Perso equilibrio al step {step_count} (era in piedi per {standing_counter} steps)")
                        standing_counter = 0

                    #CONTROLLO SUCCESSO 
                    if standing_counter >= self.config.success_threshold:
                        print(f"SUCCESSO raggiunto al step {step_count}!")

                        #CONTINUA PER 50 STEP
                        extra_steps = 0
                        while extra_steps < 50 and not (done or truncated):
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, done, truncated, _ = test_env.step(action)

                            torques = test_env.unwrapped.data.actuator_force.copy().tolist()
                            qpos = test_env.unwrapped.data.qpos.copy().tolist()
                            qvel = test_env.unwrapped.data.qvel.copy().tolist()
                            writer.writerow([step_count + extra_steps, torques, action.tolist(), qpos, qvel])

                            if extra_steps % 2 == 0:
                                try:
                                    frame = test_env.render()
                                    if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                                        frames.append(frame)
                                except Exception:
                                    pass

                            extra_steps += 1

                        return True, max_standing, step_count + extra_steps, frames
                    #FEEDBACK PERIODICO
                    if step_count % 500 == 0:
                        print(f"Step {step_count}: z_pos={stability['z_pos']:.2f}, "
                            f"standing={standing_counter}, max={max_standing}")

            print(f"Episodio {episode} terminato: max_standing={max_standing}, steps={step_count}")
            return False, max_standing, step_count, frames

        except Exception as e:
            print(f"Errore durante episodio {episode}: {e}")
            return False, 0, 0, frames

            
    def _save_video_from_frames(self, frames: list, episode: int, standing_time: int, total_steps: int):
        """
        SALVA IL VIDEO DAI FRAME RACCOLTI
        """
        if not frames:
            print("Nessun frame da salvare")
            return
        
        success_path = os.path.join(
            self.config.success_video_folder, 
            f"success_ep{episode}_standing{standing_time}steps.mp4"
        )
        
        try:
            print(f"  Salvando {len(frames)} frame in {success_path}...")
            
            with imageio.get_writer(success_path, fps=30, quality=8) as writer:
                for i, frame in enumerate(frames):
                    if frame is not None and hasattr(frame, 'shape') and frame.size > 0:
                        writer.append_data(frame)
                    
                    # FEEDBACK
                    if (i + 1) % 100 == 0:
                        print(f"Processati {i + 1}/{len(frames)} frame...")
            
            print(f"SUCCESSO! Video salvato: {success_path}")
            print(f"Manichino in piedi per {standing_time} passi consecutivi")
            print(f"Video: {len(frames)} frame, episodio: {total_steps} steps totali")
        
        except Exception as e:
            print(f"Errore nel salvataggio video: {e}")

    def create_success_video(self, model: PPO, env_manager) -> bool:
        """
        CREA UN VIDEO CHE RAPPRESENTA IL NOSTRO CONCETTO DI SUCCESSO
        """
        print("\n" + "="*60)
        print("CREAZIONE VIDEO DI SUCCESSO")
        print("="*60)
        
        test_env = None
        try:
            test_env = env_manager.create_test_env()
            
            episode = 0
            best_standing_time = 0
            best_frames = []
            best_episode_info = None
            
            while episode < self.config.max_test_episodes:
                print(f"\nTentativo {episode + 1}/{self.config.max_test_episodes}")
                
                success, max_standing, step_count, frames = self.record_success_episode(
                    model, test_env, episode
                )
                
                # AGGIORNA IL MIGLIOR TENTATIVO
                if max_standing > best_standing_time:
                    best_standing_time = max_standing
                    best_frames = frames.copy()
                    best_episode_info = (episode, step_count)
                
                if success:
                    print(f"\n EPISODIO DI SUCCESSO TROVATO!")
                    self._save_video_from_frames(frames, episode, max_standing, step_count)
                    return True
                else:
                    progress = (max_standing / self.config.success_threshold) * 100
                    print(f"   Progresso: {progress:.1f}% (target: {self.config.success_threshold} steps)")
                
                episode += 1
            
            #SALVA IL MIGLIOR TENTATIVO...
            if best_frames and best_standing_time > 10:
                print(f"\n Salvataggio miglior tentativo:")
                print(f"   Episodio {best_episode_info[0]}: {best_standing_time} steps consecutivi")
                self._save_video_from_frames(
                    best_frames, 
                    best_episode_info[0], 
                    best_standing_time, 
                    best_episode_info[1]
                )
            
            print(f"\n  Obiettivo non raggiunto dopo {self.config.max_test_episodes} episodi")
            print(f"   Miglior risultato: {best_standing_time} steps consecutivi")
            print(f"   Target richiesto: {self.config.success_threshold} steps")
            print(f"   Suggerimento: Aumenta il training o riduci success_threshold")
            
            return False
            
        except Exception as e:
            print(f"\n Errore durante creazione video: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if test_env:
                try:
                    test_env.close()
                except:
                    pass