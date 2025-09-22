"""
PROGRESSIVE CURRICULUM LEARNING WRAPPER
Con avanzamento automatico più graduale e diagnostica avanzata
"""
import gymnasium as gym
import numpy as np
from collections import deque


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum con avanzamento automatico più intelligente e soglie adattive
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # CURRICULUM PROGRESSIVO CON SOGLIE PIÙ BASSE
        self.tasks = [
            {
                'name': 'Primo movimento',
                'description': 'Qualsiasi movimento che alzi la testa da terra',
                'height_threshold': 0.32,  # Molto basso per iniziare
                'stability_threshold': 6.0,  # Meno restrittivo
                'min_episodes': 20,
                'success_target': 8,  # Ridotto
                'max_episodes': 400,
                'reward_weight': 1.0
            },
            {
                'name': 'Sollevamento iniziale',
                'description': 'Alzare la parte superiore stabilmente',
                'height_threshold': 0.45,
                'stability_threshold': 5.0,
                'min_episodes': 30,
                'success_target': 10,
                'max_episodes': 500,
                'reward_weight': 1.2
            },
            {
                'name': 'Controllo busto',
                'description': 'Mantenere il busto sollevato',
                'height_threshold': 0.60,
                'stability_threshold': 4.0,
                'min_episodes': 40,
                'success_target': 12,
                'max_episodes': 600,
                'reward_weight': 1.4
            },
            {
                'name': 'Attivazione gambe base',
                'description': 'Prime contrazioni delle gambe',
                'height_threshold': 0.75,
                'stability_threshold': 3.5,
                'min_episodes': 50,
                'success_target': 15,
                'max_episodes': 800,
                'reward_weight': 1.6
            },
            {
                'name': 'Posizione rialzata',
                'description': 'Alzarsi su ginocchia e mani',
                'height_threshold': 0.90,
                'stability_threshold': 3.0,
                'min_episodes': 60,
                'success_target': 12,
                'max_episodes': 1000,
                'reward_weight': 1.8
            },
            {
                'name': 'Pre-squat dinamico',
                'description': 'Posizione accovacciata attiva',
                'height_threshold': 1.05,
                'stability_threshold': 2.5,
                'min_episodes': 80,
                'success_target': 10,
                'max_episodes': 1200,
                'reward_weight': 2.0
            },
            {
                'name': 'Alzarsi graduale',
                'description': 'Raggiungere posizione quasi eretta',
                'height_threshold': 1.15,  # Soglia più bassa
                'stability_threshold': 2.2,
                'min_episodes': 100,
                'success_target': 8,
                'max_episodes': 1500,
                'reward_weight': 2.5
            },
            {
                'name': 'Equilibrio dinamico',
                'description': 'Stabilità eretta con movimento attivo',
                'height_threshold': 1.12,  # Leggermente più basso
                'stability_threshold': 2.0,
                'duration_threshold': 25,  # Step iniziali
                'min_episodes': 150,
                'success_target': 5,
                'max_episodes': 3000,
                'reward_weight': 3.0
            }
        ]
        
        # STATO DEL CURRICULUM
        self.current_task = 0
        self.task_success_count = 0
        self.episode_count = 0
        self.total_episodes = 0
        
        # METRICHE E TRACKING
        self.steps_standing = 0
        self.max_steps_standing = 0
        self.best_height_achieved = 0.0
        self.episode_height_history = deque(maxlen=50)
        self.episode_stability_history = deque(maxlen=50)
        
        # ADATTIVITÀ AUTOMATICA
        self.stuck_episodes = 0  # Episodi senza progressi
        self.stuck_threshold = 100  # Soglia per ridurre difficoltà
        self.difficulty_reductions = 0
        
        # TRACKING AVANZATO
        self.angular_velocity_history = deque(maxlen=20)
        self.position_history = deque(maxlen=30)
        self.action_history = deque(maxlen=15)
        
        # ANTI-FREEZING
        self.static_penalty = -1.0
        self.movement_bonus = 1.0
        self.last_joint_positions = None
        
        # PLOTTING INTEGRATION
        self._last_obs = None
        self._last_curriculum_reward = None
        self._last_original_reward = None
        self.last_info = {}
        
        # DIAGNOSTICA
        self.success_heights = []
        self.failure_reasons = {'too_low': 0, 'unstable': 0, 'fell': 0}
        self.progress_stagnation = 0
        
        print(f" Progressive Curriculum inizializzato")
        print(f" Task 1/8: {self.tasks[0]['name']}")
        print(f" Altezza target: {self.tasks[0]['height_threshold']:.2f}m")
        print(f"⚖  Stabilità max: {self.tasks[0]['stability_threshold']:.1f}")

    def reset(self, **kwargs):
        """Reset con tracking migliorato"""
        obs, info = self.env.reset(**kwargs)
        
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        
        self.episode_count += 1
        self.total_episodes += 1
        self.steps_standing = 0
        
        # Reset storia episode
        self.angular_velocity_history.clear()
        self.position_history.clear() 
        self.action_history.clear()
        
        # Tracking altezza episodio
        if len(obs) > 0:
            self.episode_height_history.append(obs[0])
        
        # Check stagnation
        if self.task_success_count == 0 and self.episode_count > 50:
            self.progress_stagnation += 1
        else:
            self.progress_stagnation = 0
            
        # Auto-reduce difficulty se bloccato
        if self.progress_stagnation > self.stuck_threshold:
            self._auto_reduce_difficulty()
            
        # Aggiorna info
        info.update({
            'curriculum_task': self.current_task,
            'task_name': self.tasks[self.current_task]['name'],
            'task_progress': f"{self.task_success_count}/{self.tasks[self.current_task]['success_target']}",
            'episode_count': self.episode_count,
            'total_episodes': self.total_episodes,
            'best_height': self.best_height_achieved,
            'current_threshold': self.tasks[self.current_task]['height_threshold'],
            'progress_stagnation': self.progress_stagnation,
            'difficulty_reductions': self.difficulty_reductions
        })
        
        self.last_info = info.copy()
        return obs, info

    def step(self, action):
        """Step con reward progressivo e diagnostica"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Tracking
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        self._last_original_reward = reward
        
        # Aggiorna storia
        if len(obs) > 0:
            self.position_history.append(obs[0])
            self.best_height_achieved = max(self.best_height_achieved, obs[0])
            
        if len(obs) > 3:
            angular_vel = np.linalg.norm(obs[1:4])
            self.angular_velocity_history.append(angular_vel)
            
        if isinstance(action, np.ndarray):
            self.action_history.append(action.copy())
        
        # Calcola curriculum reward
        curriculum_reward = self._calculate_progressive_reward(obs, reward, done)
        self._last_curriculum_reward = curriculum_reward
        
        # Check task completion
        task_completed = self._check_progressive_completion(obs, done)
        
        if task_completed:
            self._handle_task_success(obs)
        elif done and len(obs) > 0:
            self._analyze_failure(obs)
            
        # Aggiorna info
        info.update({
            'curriculum_task': self.current_task,
            'task_name': self.tasks[self.current_task]['name'],
            'task_progress': f"{self.task_success_count}/{self.tasks[self.current_task]['success_target']}",
            'original_reward': reward,
            'curriculum_reward': curriculum_reward,
            'best_height': self.best_height_achieved,
            'max_standing': self.max_steps_standing,
            'current_height': obs[0] if len(obs) > 0 else 0,
            'progress_stagnation': self.progress_stagnation
        })
        
        self.last_info = info.copy()
        return obs, curriculum_reward, done, truncated, info

    def _calculate_progressive_reward(self, obs, base_reward, done):
        """Reward shaping progressivo meno aggressivo"""
        
        if len(obs) < 1:
            return base_reward
            
        z_pos = obs[0]
        task = self.tasks[self.current_task]
        
        # Base reward moderato
        shaped_reward = base_reward * task['reward_weight']
        
        # PROGRESSIVE HEIGHT REWARD - più graduale
        height_progress = max(0, z_pos - 0.25)  # Base più bassa
        height_bonus = height_progress * (2.0 + self.current_task * 0.5)
        shaped_reward += height_bonus
        
        # STABILITY REWARD
        if len(obs) > 3:
            angular_vel = np.linalg.norm(obs[1:4])
            stability_bonus = max(0, task['stability_threshold'] - angular_vel) * 0.5
            shaped_reward += stability_bonus
            
            # Anti-freezing leggero
            if len(self.angular_velocity_history) >= 5:
                recent_movement = np.mean(list(self.angular_velocity_history)[-5:])
                if recent_movement < 0.1:  # Troppo statico
                    shaped_reward += self.static_penalty
                elif 0.3 < recent_movement < 2.0:  # Movimento controllato
                    shaped_reward += self.movement_bonus
        
        # TASK SPECIFIC BONUSES
        if self.current_task <= 2:  # Task iniziali
            # Premia qualsiasi progresso
            if z_pos > 0.3:
                shaped_reward += 2.0
            if z_pos > task['height_threshold'] * 0.8:
                shaped_reward += 3.0
                
        elif self.current_task <= 4:  # Task intermedi
            # Premia stabilità crescente
            target_height = task['height_threshold']
            if z_pos > target_height * 0.7:
                shaped_reward += 3.0
            if z_pos > target_height * 0.9:
                shaped_reward += 5.0
                
        elif self.current_task <= 6:  # Task avanzati
            # Focus su raggiungere altezza
            if z_pos > task['height_threshold']:
                shaped_reward += 8.0
            if z_pos > task['height_threshold'] * 1.1:
                shaped_reward += 12.0
                
        else:  # TASK 8 - EQUILIBRIO
            z_threshold = task['height_threshold']
            duration_target = task.get('duration_threshold', 25)
            
            if z_pos > z_threshold:
                angular_vel = np.linalg.norm(obs[1:4]) if len(obs) > 3 else 0
                if angular_vel < task['stability_threshold']:
                    self.steps_standing += 1
                    self.max_steps_standing = max(self.max_steps_standing, self.steps_standing)
                    
                    # Reward crescente per durata
                    duration_bonus = min(self.steps_standing * 0.8, duration_target * 0.8)
                    stability_bonus = max(0, 3.0 - angular_vel) * 2.0
                    
                    shaped_reward += 10.0 + duration_bonus + stability_bonus
                    
                    # Bonus milestone
                    if self.steps_standing >= duration_target // 2:
                        shaped_reward += 15.0
                    if self.steps_standing >= duration_target:
                        shaped_reward += 25.0
                else:
                    self.steps_standing = 0
            else:
                self.steps_standing = 0
        
        # PENALITÀ MODERATE
        if z_pos < 0.2:
            shaped_reward -= 3.0
        
        return shaped_reward

    def _check_progressive_completion(self, obs, done):
        """Check completion meno restrittivo"""
        
        if len(obs) < 1:
            return False
            
        z_pos = obs[0]
        task = self.tasks[self.current_task]
        
        # TASK 1-7: Altezza + stabilità
        if self.current_task < 7:
            height_ok = z_pos > task['height_threshold']
            
            if len(obs) > 3:
                angular_vel = np.linalg.norm(obs[1:4])
                stability_ok = angular_vel < task['stability_threshold']
            else:
                stability_ok = True  # Default se non abbiamo velocità angolare
                
            # Per i primi task, solo altezza è sufficiente per alcuni frame
            if self.current_task <= 2:
                return height_ok  # Meno restrittivo
            else:
                return height_ok and stability_ok
                
        else:  # TASK 8
            duration_target = task.get('duration_threshold', 25)
            return self.steps_standing >= duration_target

    def _handle_task_success(self, obs):
        """Gestisce successo task"""
        self.task_success_count += 1
        self.progress_stagnation = 0  # Reset stagnation
        
        # Salva altezza di successo per analisi
        if len(obs) > 0:
            self.success_heights.append(obs[0])
        
        task = self.tasks[self.current_task]
        
        print(f" SUCCESSO Task {self.current_task + 1}!")
        print(f"   Altezza: {obs[0]:.3f}m (target: {task['height_threshold']:.3f}m)")
        print(f"   Progresso: {self.task_success_count}/{task['success_target']}")
        
        # Check avanzamento
        should_advance = (
            self.task_success_count >= task['success_target'] and
            self.episode_count >= task['min_episodes'] and
            self.current_task < len(self.tasks) - 1
        )
        
        force_advance = (
            self.episode_count >= task['max_episodes'] and
            self.current_task < len(self.tasks) - 1
        )
        
        if should_advance or force_advance:
            self._advance_to_next_task(forced=force_advance)

    def _analyze_failure(self, obs):
        """Analizza i fallimenti per migliorare il curriculum"""
        if len(obs) < 1:
            return
            
        z_pos = obs[0]
        task = self.tasks[self.current_task]
        
        if z_pos < task['height_threshold'] * 0.5:
            self.failure_reasons['too_low'] += 1
        elif z_pos < task['height_threshold']:
            self.failure_reasons['fell'] += 1
        elif len(obs) > 3:
            angular_vel = np.linalg.norm(obs[1:4])
            if angular_vel > task['stability_threshold']:
                self.failure_reasons['unstable'] += 1

    def _auto_reduce_difficulty(self):
        """Riduce automaticamente la difficoltà se bloccato"""
        self.difficulty_reductions += 1
        task = self.tasks[self.current_task]
        
        # Riduci soglie
        old_height = task['height_threshold']
        old_stability = task['stability_threshold']
        
        task['height_threshold'] *= 0.95  # Riduci 5%
        task['stability_threshold'] *= 1.1  # Aumenta tolleranza 10%
        task['success_target'] = max(3, task['success_target'] - 1)  # Riduci target
        
        self.progress_stagnation = 0
        
        print(f" AUTO-RIDUZIONE DIFFICOLTÀ #{self.difficulty_reductions}")
        print(f"   Task: {task['name']}")
        print(f"   Altezza: {old_height:.3f} → {task['height_threshold']:.3f}")
        print(f"   Stabilità: {old_stability:.1f} → {task['stability_threshold']:.1f}")
        print(f"   Target: {task['success_target']} successi")

    def _advance_to_next_task(self, forced=False):
        """Avanza al task successivo"""
        old_task = self.current_task
        self.current_task += 1
        old_count = self.task_success_count
        self.task_success_count = 0
        self.episode_count = 0
        
        # Reset metriche
        self.best_height_achieved = 0.0
        self.max_steps_standing = 0
        self.progress_stagnation = 0
        
        status = " FORZATO" if forced else " COMPLETATO"
        
        print(f"\n{status} - AVANZAMENTO CURRICULUM!")
        print(f"    Da: Task {old_task + 1} - {self.tasks[old_task]['name']} ({old_count} successi)")
        print(f"    A:  Task {self.current_task + 1} - {self.tasks[self.current_task]['name']}")
        print(f"    Target: {self.tasks[self.current_task]['success_target']} successi")
        print(f"    Altezza target: {self.tasks[self.current_task]['height_threshold']:.3f}m")

    def get_diagnostic_info(self):
        """Informazioni diagnostiche dettagliate"""
        task = self.tasks[self.current_task]
        
        # Analisi fallimenti
        total_failures = sum(self.failure_reasons.values())
        failure_analysis = {}
        if total_failures > 0:
            for reason, count in self.failure_reasons.items():
                failure_analysis[reason] = f"{count}/{total_failures} ({count/total_failures*100:.1f}%)"
        
        # Performance recente
        recent_heights = list(self.episode_height_history)[-10:] if self.episode_height_history else []
        avg_recent_height = np.mean(recent_heights) if recent_heights else 0.0
        
        return {
            'current_task': self.current_task + 1,
            'task_name': task['name'],
            'success_count': self.task_success_count,
            'success_target': task['success_target'],
            'episode_count': self.episode_count,
            'total_episodes': self.total_episodes,
            'height_threshold': task['height_threshold'],
            'stability_threshold': task['stability_threshold'],
            'best_height': self.best_height_achieved,
            'avg_recent_height': avg_recent_height,
            'max_steps_standing': self.max_steps_standing,
            'progress_stagnation': self.progress_stagnation,
            'difficulty_reductions': self.difficulty_reductions,
            'failure_analysis': failure_analysis,
            'success_heights_avg': np.mean(self.success_heights) if self.success_heights else 0.0,
            'is_stuck': self.progress_stagnation > 30,
            'needs_difficulty_reduction': self.progress_stagnation > self.stuck_threshold
        }

    def print_diagnostic_status(self):
        """Stampa status diagnostico completo"""
        info = self.get_diagnostic_info()
        
        print(f"\n{'='*60}")
        print(f" DIAGNOSTIC STATUS - Task {info['current_task']}/8")
        print(f" {info['task_name']}")
        print(f"{'='*60}")
        
        print(f" PROGRESSO:")
        print(f"   Successi: {info['success_count']}/{info['success_target']}")
        print(f"   Episodi: {info['episode_count']} (totali: {info['total_episodes']})")
        print(f"   Stagnazione: {info['progress_stagnation']} {'⚠' if info['is_stuck'] else ''}")
        
        print(f" ALTEZZE:")
        print(f"   Target: {info['height_threshold']:.3f}m")
        print(f"   Best: {info['best_height']:.3f}m")
        print(f"   Media recente: {info['avg_recent_height']:.3f}m")
        
        if info['failure_analysis']:
            print(f" ANALISI FALLIMENTI:")
            for reason, stat in info['failure_analysis'].items():
                print(f"   {reason}: {stat}")
        
        if info['difficulty_reductions'] > 0:
            print(f"🔧 Riduzioni difficoltà: {info['difficulty_reductions']}")
        
        print(f"{'='*60}\n")

    def get_wrapper_attr(self, name):
        """Accesso agli attributi del wrapper"""
        if hasattr(self, name):
            return getattr(self, name)
        return None


# Aliases
CurriculumWrapper = CurriculumWrapper
