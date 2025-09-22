"""
MODULO PER LA GENERAZIONE DI GRAFICI DI TRAINING - VERSIONE ROBUSTA
Gestisce la creazione di grafici per reward medi e giunti dell'umanoide
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
from config import CONFIG
import pandas as pd
from collections import deque
import seaborn as sns

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_palette("husl")

def safe_convert_to_python_type(obj: Any) -> Any:
    """
    CONVERTE QUALSIASI TIPO NUMPY IN TIPO PYTHON STANDARD PER JSON
    """
    if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [safe_convert_to_python_type(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {key: safe_convert_to_python_type(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_convert_to_python_type(x) for x in obj]
    else:
        return obj

class PlottingManager:
    """
    GESTISCE LA RACCOLTA DATI E GENERAZIONE GRAFICI DURANTE IL TRAINING
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.plots_folder = "plots"
        self.data_folder = "training_data"
        
        # Dati per i grafici - SEMPRE tipi Python standard
        self.reward_history: List[float] = []
        self.joint_history: List[Dict[str, Any]] = []
        self.timesteps_history: List[int] = []
        
        # Buffer per calcoli moving average
        self.reward_buffer = deque(maxlen=100)
        
        self._setup_folders()
        
    def _setup_folders(self):
        """CREA LE CARTELLE NECESSARIE E CARICA DATI ESISTENTI"""
        os.makedirs(self.plots_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        self._load_existing_data()
        
    def log_reward_data(self, timestep: Union[int, np.integer], mean_reward: Union[float, np.floating]):
        """
        REGISTRA I DATI DEI REWARD - CONVERSIONE SICURA
        """
        # Conversione esplicita e sicura
        timestep_safe = int(timestep)
        reward_safe = float(mean_reward)
        
        self.timesteps_history.append(timestep_safe)
        self.reward_history.append(reward_safe)
        self.reward_buffer.append(reward_safe)
        
        # Salva periodicamente
        if len(self.reward_history) % 10 == 0:
            self._save_reward_data()
    
    def log_joint_data(self, timestep: Union[int, np.integer], 
                      joint_positions: np.ndarray, joint_velocities: np.ndarray):
        """
        REGISTRA I DATI DEI GIUNTI - CONVERSIONE SICURA
        """
        joint_data = {
            'timestep': int(timestep),
            'positions': [float(x) for x in joint_positions.flatten()],
            'velocities': [float(x) for x in joint_velocities.flatten()],
            'timestamp': datetime.now().isoformat()
        }
        self.joint_history.append(joint_data)
        
        # Salva periodicamente per evitare perdita dati
        if len(self.joint_history) % 50 == 0:
            self._save_joint_data()
    
    def _save_reward_data(self):
        """SALVA I DATI DEI REWARD SU FILE - VERSIONE SICURA"""
        try:
            # Assicura che TUTTI i dati siano tipi Python standard
            safe_data = {
                'timesteps': [int(x) for x in self.timesteps_history],
                'rewards': [float(x) for x in self.reward_history],
                'last_updated': datetime.now().isoformat()
            }
            
            filepath = os.path.join(self.data_folder, 'reward_data.json')
            with open(filepath, 'w') as f:
                json.dump(safe_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Impossibile salvare reward data: {e}")
    
    def _save_joint_data(self):
        """SALVA I DATI DEI GIUNTI SU FILE - VERSIONE SICURA"""
        try:
            # Pulisce ricorsivamente tutti i dati
            safe_data = safe_convert_to_python_type(self.joint_history)
            
            filepath = os.path.join(self.data_folder, 'joint_data.json')
            with open(filepath, 'w') as f:
                json.dump(safe_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Impossibile salvare joint data: {e}")
    
    def _load_existing_data(self):
        """CARICA DATI ESISTENTI SE DISPONIBILI - CONVERSIONE SICURA"""
        # Carica reward data
        reward_file = os.path.join(self.data_folder, 'reward_data.json')
        if os.path.exists(reward_file):
            try:
                with open(reward_file, 'r') as f:
                    data = json.load(f)
                    # Conversione esplicita e sicura
                    self.timesteps_history = [int(float(x)) for x in data.get('timesteps', [])]
                    self.reward_history = [float(x) for x in data.get('rewards', [])]
                    print(f"Caricati {len(self.reward_history)} punti dati reward")
            except Exception as e:
                print(f"Errore caricamento reward data: {e}")
                self.timesteps_history = []
                self.reward_history = []
        
        # Carica joint data
        joint_file = os.path.join(self.data_folder, 'joint_data.json')
        if os.path.exists(joint_file):
            try:
                with open(joint_file, 'r') as f:
                    raw_data = json.load(f)
                    self.joint_history = safe_convert_to_python_type(raw_data)
                    print(f"Caricati {len(self.joint_history)} punti dati joint")
            except Exception as e:
                print(f"Errore caricamento joint data: {e}")
                self.joint_history = []
    
    def create_reward_plot(self, save: bool = True, show: bool = False) -> Optional[str]:
        """CREA GRAFICO DEI REWARD MEDI NEL TEMPO"""
        if not self.reward_history:
            print("Nessun dato reward disponibile per il plot")
            return None
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Grafico principale - Reward nel tempo
            ax1.plot(self.timesteps_history, self.reward_history, 
                    alpha=0.6, linewidth=0.8, label='Reward istantaneo', color='lightblue')
            
            # Moving average per smoothing
            if len(self.reward_history) > 10:
                window_size = min(50, len(self.reward_history) // 10)
                rewards_smooth = pd.Series(self.reward_history).rolling(window=window_size, center=True).mean()
                ax1.plot(self.timesteps_history, rewards_smooth, 
                        linewidth=2, label=f'Media mobile ({window_size} punti)', color='darkblue')
            
            ax1.set_xlabel('Timesteps', fontsize=14)
            ax1.set_ylabel('Reward Medio', fontsize=14)
            ax1.set_title(f'Evoluzione Reward Medi Durante Training\nUltimo reward: {self.reward_history[-1]:.2f}', fontsize=16)
            ax1.grid(False)
            ax1.legend()
            
            # Subplot - Istogramma distribuzione reward
            ax2.hist(self.reward_history, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            mean_reward = float(np.mean(self.reward_history))
            median_reward = float(np.median(self.reward_history))
            
            ax2.axvline(mean_reward, color='red', linestyle='--', label=f'Media: {mean_reward:.2f}')
            ax2.axvline(median_reward, color='green', linestyle='--', label=f'Mediana: {median_reward:.2f}')
            ax2.set_xlabel('Reward', fontsize=14)
            ax2.set_ylabel('Frequenza', fontsize=14)
            ax2.set_title('Distribuzione dei Reward', fontsize=16)
            ax2.legend()
            ax2.grid(False)
            
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reward_plot_{timestamp}.png"
                filepath = os.path.join(self.plots_folder, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Grafico reward salvato: {filepath}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
            return filepath if save else None
            
        except Exception as e:
            print(f"Errore nella creazione grafico reward: {e}")
            return None
    
    def create_joints_plot(self, save: bool = True, show: bool = False, 
                          max_joints: int = 10) -> List[str]:
        """CREA GRAFICI SEPARATI DEI GIUNTI DELL'UMANOIDE NEL TEMPO"""
        if not self.joint_history:
            print("Nessun dato joint disponibile per il plot")
            return []
        
        generated_plots = []
        
        try:
            # Estrai dati
            timesteps = [data['timestep'] for data in self.joint_history]
            positions = np.array([data['positions'] for data in self.joint_history])
            velocities = np.array([data['velocities'] for data in self.joint_history])
            
            n_joints = min(max_joints, positions.shape[1])
            
            # 1. GRAFICO POSIZIONI GIUNTI
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            for i in range(n_joints):
                ax1.plot(timesteps, positions[:, i], alpha=0.7, linewidth=1.5, 
                        label=f'Joint {i+1}')
            ax1.set_xlabel('Timesteps', fontsize=14)
            ax1.set_ylabel('Posizione (rad)', fontsize=14)
            ax1.set_title(f'Posizioni Giunti nel Tempo (primi {n_joints} giunti)', fontsize=16)
            ax1.grid(False)
            ax1.legend()
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename1 = f"joints_positions_{timestamp}.png"
                filepath1 = os.path.join(self.plots_folder, filename1)
                plt.savefig(filepath1, dpi=300, bbox_inches='tight')
                generated_plots.append(filepath1)
                print(f"Grafico posizioni giunti salvato: {filepath1}")
                
            if show:
                plt.show()
            else:
                plt.close()
            
            # 2. GRAFICO VELOCITÀ GIUNTI
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            for i in range(n_joints):
                ax2.plot(timesteps, velocities[:, i], alpha=0.7, linewidth=1.5,
                        label=f'Joint {i+1}')
            ax2.set_xlabel('Timesteps', fontsize=14)
            ax2.set_ylabel('Velocità (rad/s)', fontsize=14)
            ax2.set_title(f'Velocità Giunti nel Tempo (primi {n_joints} giunti)', fontsize=16)
            ax2.grid(False)
            ax2.legend()
            plt.tight_layout()
            
            if save:
                filename2 = f"joints_velocities_{timestamp}.png"
                filepath2 = os.path.join(self.plots_folder, filename2)
                plt.savefig(filepath2, dpi=300, bbox_inches='tight')
                generated_plots.append(filepath2)
                print(f"Grafico velocità giunti salvato: {filepath2}")
                
            if show:
                plt.show()
            else:
                plt.close()
            
            # 3. HEATMAP CORRELAZIONI POSIZIONI
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            pos_subset = positions[:, :min(8, positions.shape[1])]
            correlation_pos = np.corrcoef(pos_subset.T)
            im = ax3.imshow(correlation_pos, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_title('Correlazione Posizioni Giunti', fontsize=16)
            ax3.set_xlabel('Joint Index', fontsize=14)
            ax3.set_ylabel('Joint Index', fontsize=14)
            plt.colorbar(im, ax=ax3)
            plt.tight_layout()
            
            if save:
                filename3 = f"joints_correlation_{timestamp}.png"
                filepath3 = os.path.join(self.plots_folder, filename3)
                plt.savefig(filepath3, dpi=300, bbox_inches='tight')
                generated_plots.append(filepath3)
                print(f"Grafico correlazioni giunti salvato: {filepath3}")
                
            if show:
                plt.show()
            else:
                plt.close()
            
            # 4. RANGE DI MOVIMENTO PER GIUNTO
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            joint_ranges = []
            
            for i in range(min(12, positions.shape[1])):
                pos_range = float(np.max(positions[:, i]) - np.min(positions[:, i]))
                joint_ranges.append(pos_range)
            
            x_joints = range(len(joint_ranges))
            bars = ax4.bar(x_joints, joint_ranges, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('Joint Index', fontsize=14)
            ax4.set_ylabel('Range di Movimento (rad)', fontsize=14)
            ax4.set_title('Range di Movimento per Giunto', fontsize=16)
            ax4.grid(False)
            
            # Aggiungi valori sulle barre
            max_range = max(joint_ranges) if joint_ranges else 1.0
            for i, (bar, range_val) in enumerate(zip(bars, joint_ranges)):
                if range_val > max_range * 0.1:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{range_val:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save:
                filename4 = f"joints_range_{timestamp}.png"
                filepath4 = os.path.join(self.plots_folder, filename4)
                plt.savefig(filepath4, dpi=300, bbox_inches='tight')
                generated_plots.append(filepath4)
                print(f"Grafico range giunti salvato: {filepath4}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
            return generated_plots
            
        except Exception as e:
            print(f"Errore nella creazione grafici giunti: {e}")
            return generated_plots
    
    def create_combined_summary_plot(self, save: bool = True, show: bool = False) -> Optional[str]:
        """CREA UN GRAFICO RIASSUNTIVO COMBINATO"""
        if not self.reward_history and not self.joint_history:
            print("Nessun dato disponibile per il plot combinato")
            return None
            
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Reward plot (grande)
            if self.reward_history:
                ax1 = fig.add_subplot(gs[0, :2])
                ax1.plot(self.timesteps_history, self.reward_history, alpha=0.6, linewidth=1)
                
                if len(self.reward_history) > 10:
                    window_size = min(50, len(self.reward_history) // 10)
                    rewards_smooth = pd.Series(self.reward_history).rolling(window=window_size).mean()
                    ax1.plot(self.timesteps_history, rewards_smooth, linewidth=2, color='red')
                    
                ax1.set_xlabel('Timesteps', fontsize=14)
                ax1.set_ylabel('Reward Medio', fontsize=14)
                ax1.set_title('Evoluzione Reward', fontsize=16)
                ax1.grid(False)
            
            # Statistiche reward
            if self.reward_history:
                ax2 = fig.add_subplot(gs[0, 2])
                mean_r = float(np.mean(self.reward_history))
                median_r = float(np.median(self.reward_history))
                std_r = float(np.std(self.reward_history))
                min_r = float(np.min(self.reward_history))
                max_r = float(np.max(self.reward_history))
                last_r = float(self.reward_history[-1])
                
                trend_symbol = '↗' if len(self.reward_history) > 50 and last_r > mean_r else '→'
                
                stats_text = f"""Statistiche Reward:
                
Media: {mean_r:.2f}
Mediana: {median_r:.2f}
Std: {std_r:.2f}
Min: {min_r:.2f}
Max: {max_r:.2f}

Ultimo: {last_r:.2f}
Trend: {trend_symbol}"""
                
                ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
            
            # Joint dynamics (se disponibili)
            if self.joint_history:
                ax3 = fig.add_subplot(gs[1, :])
                recent_data = self.joint_history[-100:]  # Ultimi 100 punti
                timesteps = [data['timestep'] for data in recent_data]
                positions = np.array([data['positions'] for data in recent_data])
                
                # Plot solo alcuni giunti chiave per chiarezza
                key_joints = min(6, positions.shape[1])
                for i in range(key_joints):
                    ax3.plot(timesteps, positions[:, i], alpha=0.7, linewidth=1, label=f'Joint {i+1}')
                    
                ax3.set_xlabel('Timesteps (ultimi 100 campioni)', fontsize=14)
                ax3.set_ylabel('Posizione Giunti (rad)', fontsize=14)
                ax3.set_title('Dinamica Giunti Recente', fontsize=16)
                ax3.legend(ncol=3)
                ax3.grid(False)
            
            plt.suptitle(f'Riassunto Training - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=18, fontweight='bold')
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_summary_{timestamp}.png"
                filepath = os.path.join(self.plots_folder, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Grafico riassuntivo salvato: {filepath}")
                
            if show:
                plt.show()
            else:
                plt.close()
                
            return filepath if save else None
            
        except Exception as e:
            print(f"Errore nella creazione grafico riassuntivo: {e}")
            return None
    
    def generate_all_plots(self, show: bool = False) -> List[str]:
        """GENERA TUTTI I GRAFICI DISPONIBILI"""
        print("\n" + "="*50)
        print("GENERAZIONE GRAFICI DI TRAINING")
        print("="*50)
        
        generated_plots = []
        
        # Salva tutti i dati prima di generare i plot (versione sicura)
        try:
            self._save_reward_data()
            self._save_joint_data()
        except Exception as e:
            print(f"Warning: Errore nel salvataggio dati: {e}")
        
        # Genera reward plot
        try:
            reward_plot = self.create_reward_plot(save=True, show=show)
            if reward_plot:
                generated_plots.append(reward_plot)
        except Exception as e:
            print(f"Errore generazione grafico reward: {e}")
        
        # Genera joint plots (ora multipli)
        try:
            joint_plots = self.create_joints_plot(save=True, show=show)
            generated_plots.extend(joint_plots)
        except Exception as e:
            print(f"Errore generazione grafici giunti: {e}")
        
        # Genera summary plot
        try:
            summary_plot = self.create_combined_summary_plot(save=True, show=show)
            if summary_plot:
                generated_plots.append(summary_plot)
        except Exception as e:
            print(f"Errore generazione grafico riassuntivo: {e}")
        
        print(f"\nGenerati {len(generated_plots)} grafici:")
        for plot in generated_plots:
            print(f"  - {plot}")
        
        return generated_plots
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - salva tutti i dati in modo sicuro"""
        try:
            self._save_reward_data()
            self._save_joint_data()
        except Exception as e:
            print(f"Warning: Errore durante salvataggio finale: {e}")