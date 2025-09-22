"""
CURRICULUM LEARNING WRAPPER - IMPLEMENTA UN APPRENDIMENTO GRADUALE E DETTAGLIATO
CON INTEGRAZIONE PLOTTING MANAGER
"""
import gymnasium as gym
import numpy as np


class ImprovedCurriculumWrapper(gym.Wrapper):
    """
    WRAPPER CHE IMPLEMENTA UN CURRICULUM LEARNING PER L'AMBIENTE HUMANOID
    CON TRACKING INTEGRATO PER IL PLOTTING SYSTEM
    
    Il curriculum è diviso in 8 sotto-task progressivi e specifici:
    1. Controllo testa e collo - Stabilizzare la parte superiore
    2. Sollevamento torso - Alzare il busto da terra
    3. Attivazione gambe - Prime contrazioni muscolari delle gambe
    4. Posizione pre-accovacciamento - Alzarsi su mani e ginocchia
    5. Posizione accovacciata - Portarsi in squat profondo
    6. Preparazione alzata - Posizionare piedi e preparare spinta
    7. Alzarsi in piedi - Estendere gambe e raggiungere posizione eretta
    8. Equilibrio stabile - Mantenere la posizione per periodi crescenti
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        #DEFINIZIONE  DEI TASK DEL CURRICULUM
        self.tasks = [
            {
                'name': 'Controllo testa e collo',
                'description': 'Stabilizzare la parte superiore del corpo',
                'reward_weight': 0.8,
                'success_threshold': 0.4,  #ALTEZZA MINIMA TESTA
                'min_episodes': 50,        #MINIMO EPISODI PER AVANZARE
                'success_target': 15,      #SUCCESSI NECESSARI
                'max_episodes': 800       #MASSIMO EPISODI
            },
            {
                'name': 'Sollevamento torso',
                'description': 'Alzare il busto da terra e mantenerlo stabile',
                'reward_weight': 1.0,
                'success_threshold': 0.6,
                'min_episodes': 80,
                'success_target': 20,
                'max_episodes': 1000
            },
            {
                'name': 'Attivazione gambe',
                'description': 'Prime contrazioni muscolari delle gambe',
                'reward_weight': 1.1,
                'success_threshold': 0.7,  #ALTEZZA CON I PRIMI MOVIMENTI GAMBE
                'min_episodes': 100,
                'success_target': 25,
                'max_episodes': 1200
            },
            {
                'name': 'Posizione pre-accovacciamento',
                'description': 'preparazione all accovaciamento',
                'reward_weight': 1.3,
                'success_threshold': 0.9,
                'min_episodes': 120,
                'success_target': 20,
                'max_episodes': 1500
            },
            {
                'name': 'Posizione accovacciata',
                'description': 'Posizione accovacciata stabile',
                'reward_weight': 1.5,
                'success_threshold': 1.1,
                'min_episodes': 150,
                'success_target': 15,
                'max_episodes': 1800
            },
            {
                'name': 'Preparazione alzata',
                'description': 'Posizionare piedi e preparare la spinta',
                'reward_weight': 1.7,
                'success_threshold': 1.15,
                'min_episodes': 100,
                'success_target': 12,
                'max_episodes': 1500
            },
            {
                'name': 'Alzarsi in piedi',
                'description': 'Estendere gambe e raggiungere posizione eretta',
                'reward_weight': 2.0,
                'success_threshold': 1.17,  #ABBASSATA DA 1.4 A 1.17
                'min_episodes': 200,
                'success_target': 8,  #RIDOTTO DA 10 A 8
                'max_episodes': 2000
            },
            {
                'name': 'Equilibrio stabile',
                'description': 'Mantenere equilibrio robusto con controllo oscillazioni',
                'reward_weight': 3.0,  #Aumentato da 2.5
                'success_threshold': 40,  #Aumentato da 25 step consecutivi
                'min_episodes': 100,
                'success_target': 5,  #Aumentato da 3 a 5 per più robustezza
                'max_episodes': 6000
            }
        ]
        
        #STATO DEL CURRICULUM
        self.current_task = 0
        self.task_success_count = 0
        self.episode_count = 0
        
        #METRICHE DETTAGLIATE
        self.steps_standing = 0
        self.max_steps_standing = 0
        self.best_height_achieved = 0.0
        self.stability_history = []
        
        #PARAMETRI DINAMICI PER TASK 8
        self.dynamic_threshold = 25  #RIDOTTO DA 30 A 25 - SOGLIA DINAMICA PER L'ULTIMO TASK
        self.threshold_increases = 0  #NUMERO DI INCREMENTI DELLA SOGLIA
        
        # TRACCIAMENTO PER PLOTTING SYSTEM
        self._last_obs = None
        self._last_curriculum_reward = None
        self._last_original_reward = None
        self.last_info = {}
        
        print(f" Curriculum Learning con Plotting inizializzato")
        print(f" Task 1/8: {self.tasks[0]['name']}")
        print(f" Target: {self.tasks[0]['success_target']} successi")
        
        #Nuovi parametri per stabilità avanzata
        self.stability_window = 10  #Finestra per calcolare stabilità 
        self.angular_velocity_history = []
        self.position_history = []
        self.oscillation_penalty_factor = 2.0
        
        #Contatori specifici per Task 8
        self.consecutive_stable_episodes = 0
        self.total_falls = 0
        self.stability_score_history = []
    

    def _calculate_stability_score(self):
        """Calcola un punteggio di stabilità basato su variazioni recenti"""
        if len(self.angular_velocity_history) < 5:
            return 0
        
        # Stabilità angolare (bassa varianza è meglio)
        angular_variance = np.var(self.angular_velocity_history)
        angular_stability = max(0, 1.0 - angular_variance)
        
        # Stabilità posizionale (piccole variazioni in altezza)
        if len(self.position_history) >= 5:
            position_variance = np.var(self.position_history)
            position_stability = max(0, 1.0 - position_variance * 10)
        else:
            position_stability = 0
        
        # Stabilità media delle velocità 
        mean_angular_vel = np.mean(self.angular_velocity_history)
        velocity_stability = max(0, 1.0 - mean_angular_vel / 2.0)
        
        return (angular_stability + position_stability + velocity_stability) / 3.0
    

    def _calculate_oscillation_penalty(self):
        """Penalizza oscillazioni eccessive"""
        history = self.angular_velocity_history[-20:]

        if len(history) < 5:
            return 0.0

        changes = 0
        threshold = 0.05  #IGNORA MICRO-OSCILLAZIONI

        for i in range(2, len(history)):
            diff = history[i] - history[i-1]
            prev_diff = history[i-1] - history[i-2]

            #CONSIDERA L'INVERSIONE SOLO SE L'OSCILLAZIONE È SIGNIFICATIVA
            if abs(diff) > threshold and abs(prev_diff) > threshold:
                if diff * prev_diff < 0:
                    changes += 1

        #PENALITÀ PROPORZIONALE MA LIMITATA
        oscillation_penalty = changes * self.oscillation_penalty_factor * 0.5
        return min(oscillation_penalty, 5.0)

    def reset(self, **kwargs):
        """RESET DELL'AMBIENTE CON GESTIONE DEL CURRICULUM E TRACKING"""
        obs, info = self.env.reset(**kwargs)
        
        # TRACKING PER PLOTTING
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        
        self.episode_count += 1
        self.steps_standing = 0
        
        if self.steps_standing == 0:
            self.consecutive_stable_episodes += 1
        else:
            self.consecutive_stable_episodes = 0
        # Reset history
        self.angular_velocity_history = []
        self.position_history = []
        self.steps_standing = 0
        #RESET DEI CONTATORI SPECIFICI PER TASK
        if hasattr(self, '_task0_stable_steps'):
            self._task0_stable_steps = 0
        
        #AGGIUNGI ALTEZZA CORRENTE ALLA STABILITY HISTORY
        if len(obs) > 0:
            self.stability_history.append(obs[0])
            # MANTIENI SOLO GLI ULTIMI 50 STEP
            if len(self.stability_history) > 50:
                self.stability_history = self.stability_history[-50:]
        
        #AGGIORNA LE INFO CON TASK CORRENTE
        info.update({
            'curriculum_task': self.current_task,
            'task_name': self.tasks[self.current_task]['name'],
            'task_progress': self.task_success_count,
            'task_target': self.tasks[self.current_task]['success_target'],
            'episode_count': self.episode_count,
            'consecutive_stable_episodes': self.consecutive_stable_episodes,
            'total_falls': self.total_falls,
            'stability_score': self._calculate_stability_score()
        })
        
        # SALVA INFO PER CALLBACK ACCESS
        self.last_info = info.copy()
        
        return obs, info
    
    def step(self, action):
        """STEP CON REWARD SHAPING E TRACKING INTEGRATO"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # TRACKING PER PLOTTING - SALVA ULTIMA OSSERVAZIONE
        self._last_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        self._last_original_reward = reward
        
        #AGGIORNA STABILITY HISTORY AD OGNI STEP
        if len(obs) > 0:
            self.stability_history.append(obs[0])
            if len(self.stability_history) > 50:
                self.stability_history = self.stability_history[-50:]
        
        #CALCOLA IL REWARD BASATO SUL TASK CORRENTE
        curriculum_reward = self._calculate_detailed_curriculum_reward(obs, reward, done)
        
        # SALVA IL REWARD DEL CURRICULUM PER TRACKING
        self._last_curriculum_reward = curriculum_reward
        
        #CONTROLLA SE IL TASK È COMPLETATO
        task_completed = self._check_detailed_task_completion(obs, done)
        
        #AGGIORNA IL CURRICULUM SE NECESSARIO
        if task_completed:
            self._update_curriculum_progress()
        
        #AGGIORNA LE INFO
        info.update({
            'curriculum_task': self.current_task,
            'task_name': self.tasks[self.current_task]['name'],
            'task_progress': f"{self.task_success_count}/{self.tasks[self.current_task]['success_target']}",
            'original_reward': reward,
            'curriculum_reward': curriculum_reward,
            'best_height': self.best_height_achieved,
            'max_standing': self.max_steps_standing
        })
        
        # SALVA INFO PER CALLBACK ACCESS
        self.last_info = info.copy()
        
        return obs, curriculum_reward, done, truncated, info
    
    def _calculate_detailed_curriculum_reward(self, obs, base_reward, done):
        """CALCOLA IL REWARD PER OGNI TASK CON TRACKING"""
        
        if len(obs) < 6:
            return base_reward
        
        #ESTRAI INFORMAZIONI DETTAGLIATE
        z_pos = obs[0]  #ALTEZZA DEL CENTRO DI MASSA
        angular_vel = np.linalg.norm(obs[1:4])  #VELOCITÀ ANGOLARE
        linear_vel = np.linalg.norm(obs[4:7]) if len(obs) > 6 else 0 #VELOCITÀ LINEARE
        
        #AGGIORNA IL RECORD DI ALTEZZA 
        self.best_height_achieved = max(self.best_height_achieved, z_pos)
        
        #AGGIORNA LA HISTORY PER ANALISI STABILITÀ
        self.angular_velocity_history.append(angular_vel)
        self.position_history.append(z_pos)
        
        #MANTIENI SOLO GLI ULTIMI N  VALORI
        if len(self.angular_velocity_history) > self.stability_window:
            self.angular_velocity_history = self.angular_velocity_history[-self.stability_window:]
            self.position_history = self.position_history[-self.stability_window:]

        task = self.tasks[self.current_task]
        shaped_reward = base_reward * task['reward_weight']
        
        #REWARD SHAPING SPECIFICO E PROGRESSIVO
        if self.current_task == 0:  #CONTROLLO TESTA E COLLO
            #FOCUS SU STABILITÀ E CONTROLLO
            head_stability = max(0, z_pos - 0.3) * 1.5
            angular_penalty = -min(angular_vel, 5.0) * 0.05
            progress_bonus = 0.5 if z_pos > 0.35 else 0
            shaped_reward += head_stability + angular_penalty + progress_bonus
            
        elif self.current_task == 1:  #SOLLEVAMENTO TORSO
            #REWARD PER ALZARE GRADUALMENTE IL BUSTO
            torso_height = max(0, z_pos - 0.4) * 2.0
            stability_bonus = max(0, 2.0 - angular_vel) * 0.1
            consistent_height = 1.0 if z_pos > 0.55 else 0
            shaped_reward += torso_height + stability_bonus + consistent_height
            
        elif self.current_task == 2:  #ATTIVAZIONE GAMBE
            #REWARD PER I PRIMI MOVIMENTI DELLE GAMBE
            leg_activation = max(0, z_pos - 0.5) * 2.5
            movement_control = max(0, 3.0 - linear_vel) * 0.1  #MOVIMENTI CONTROLLATI
            height_progress = 2.0 if z_pos > 0.65 else 0
            shaped_reward += leg_activation + movement_control + height_progress
            
        elif self.current_task == 3:  #POSIZIONE QUADRUPEDE
            #REWARD PER POSIZIONARSI SU GAMBE E GINOCCHIA
            quadruped_height = max(0, z_pos - 0.6) * 3.0
            stability_bonus = max(0, 2.5 - angular_vel) * 0.2
            target_zone = 3.0 if 0.8 <= z_pos <= 1.0 else 0
            shaped_reward += quadruped_height + stability_bonus + target_zone
            
        elif self.current_task == 4:  #POSIZIONE ACCOVACCIATA
            #REWARD PER SQUAT STABILE
            squat_position = max(0, z_pos - 0.8) * 3.5
            balance_bonus = max(0, 2.0 - angular_vel) * 0.3
            squat_zone = 4.0 if 1.0 <= z_pos <= 1.2 else 0
            shaped_reward += squat_position + balance_bonus + squat_zone
            
        elif self.current_task == 5:  #PREPARAZIONE ALZATA
            #REWARD PER PREPARAZIONE ALLA SPINTA FINALE
            prep_height = max(0, z_pos - 1.0) * 4.0
            readiness_bonus = max(0, 1.5 - angular_vel) * 0.4
            launch_zone = 5.0 if 1.15 <= z_pos <= 1.35 else 0
            shaped_reward += prep_height + readiness_bonus + launch_zone
            
        elif self.current_task == 6:  #ALZARSI IN PIEDI
            #REWARD PER RAGGIUNGERE LA POSIZIONE ERETTA
            standing_height = max(0, z_pos - 1.0) * 5.0  # PARTENZA PIÙ BASSA
            uprightness = max(0, 1.0 - angular_vel) * 0.5
            standing_zone = 8.0 if z_pos > 1.15 else 0  # SOGLIA PIÙ BASSA
            shaped_reward += standing_height + uprightness + standing_zone
            
        elif self.current_task == 7:  #TASK 8 - EQUILIBRIO STABILE 
            
            #CONDIZIONI  PER LA STABILITÀ
            is_standing = z_pos > 1.05  
            is_balanced = angular_vel < 1.5  
            is_controlled = linear_vel < 0.8  
            
            if is_standing and is_balanced and is_controlled:
                self.steps_standing += 1
                self.max_steps_standing = max(self.max_steps_standing, self.steps_standing)
                
                #1. REWARD BASE PER STARE IN PIEDI
                base_standing_reward = 5.0
                
                #2. REWARD PROGRESSIVO PER DURATA
                duration_multiplier = min(self.steps_standing * 0.4, 20.0)
                
                #3. STABILITA COMPOSITA BASATA SU HISTORY
                stability_score = self._calculate_stability_score()
                stability_bonus = stability_score * 3.0
                
                #4. PENALITÀ ANTI OSCILLAZIONE
                oscillation_penalty = self._calculate_oscillation_penalty()
                
                #5. PESO BONUS
                optimal_height = 1.17  #ALTEZZA OTTIMALE
                height_deviation = abs(z_pos - optimal_height)
                height_bonus = max(0, 2.0 - height_deviation * 5.0)
                
                #6. NUOVO: BONUS CONSISTENZA PER LUNGHI PERIODI 
                consistency_bonus = 0
                if self.steps_standing >= 20:
                    consistency_bonus += 5.0
                if self.steps_standing >= 35:
                    consistency_bonus += 10.0
                if self.steps_standing >= 50:
                    consistency_bonus += 15.0
                
                #7. NUOVO: BONUS PER LA STABILITÀ DELLA VELOCITÀ
                velocity_stability = max(0, 2.0 - angular_vel) * 2.0
                
                total_standing_reward = (
                    base_standing_reward + 
                    duration_multiplier + 
                    stability_bonus + 
                    height_bonus + 
                    consistency_bonus + 
                    velocity_stability - 
                    oscillation_penalty
                )
                
                shaped_reward += total_standing_reward
            else:
                #PENALITÀ PIÙ SEVERE PER PERDITA EQUILIBRIO
                if self.steps_standing > 0:
                    #PENALITÀ PROPORZIONALE A QUANTI STEP È STATO IN PIEDI
                    fall_penalty = min(self.steps_standing * 0.2, 10.0)
                    shaped_reward -= fall_penalty
                    self.total_falls += 1
                    
                self.steps_standing = 0
                
                #PENALITÀ PER POSIZIONI PERICOLOSE
                if z_pos < 0.9:
                    shaped_reward -= 3.0
                if angular_vel > 4.0:
                    shaped_reward -= 2.0
                    
            #BONUS/PENALITÀ GLOBALI PER IL TASK 8
            
            #PENALITÀ PER ALTA FREQUENZA CADUTE
            if self.total_falls > 0 and hasattr(self, 'episode_count'):
                fall_rate = self.total_falls / max(self.episode_count, 1)
                if fall_rate > 0.8:  #PIÙ DELL' 80% DEGLI EPISODI
                    shaped_reward -= 1.0
            
            #BONUS PER EPISODI SENZA CADUTE
            if not done and z_pos > 1.0:
                shaped_reward += 0.5
        
        #BONUS GLOBALI
        #BONUS PER EPISODI LUNGHI SENZA CADUTE
        if not done or z_pos > 0.5:
            shaped_reward += 0.1
        
        #PENALITÀ PER CADUTE GRAVI
        if z_pos < 0.3:
            shaped_reward -= 2.0
        #RECORD ALTEZZA
        if z_pos > self.best_height_achieved * 0.95:
            shaped_reward += 0.5
        return shaped_reward
    
    def _check_detailed_task_completion(self, obs, done):
        """CONTROLLO DEL COMPLETAMENTO DEL TASK"""
        
        if len(obs) < 1:
            return False
        
        z_pos = obs[0]
        angular_vel = np.linalg.norm(obs[1:4]) if len(obs) > 3 else 0
        
        task = self.tasks[self.current_task]
        
        #CONDIZIONI SPECIFICHE PER OGNI TASK - PIÙ RESTRITTIVE
        if self.current_task == 0:  #CONTROLLO TESTA E COLLO
            # DEVE ESSERE STABILE PER ALMENO 20 STEP CONSECUTIVI
            if (z_pos > task['success_threshold'] and angular_vel < 4.0):
                if not hasattr(self, '_task0_stable_steps'):
                    self._task0_stable_steps = 0
                self._task0_stable_steps += 1
                return self._task0_stable_steps >= 20
            else:
                self._task0_stable_steps = 0
                return False
            
        elif self.current_task == 1:  #SOLLEVAMENTO TORSO
            # CONTROLLA GLI ULTIMI 30 STEP DELLA HISTORY
            if len(self.stability_history) < 30:
                return False
            recent_history = self.stability_history[-30:]
            stable_frames = sum(1 for h in recent_history if h > 0.55)
            return (z_pos > task['success_threshold'] and 
                   angular_vel < 3.5 and 
                   stable_frames >= 25)  #25 SU 30 DEVONO ESSERE STABILI
            
        elif self.current_task == 2:  #ATTIVAZIONE GAMBE
            return (z_pos > task['success_threshold'] and 
                   angular_vel < 3.0)
            
        elif self.current_task == 3:  # POSIZIONE QUADRUPEDE
            return (0.8 <= z_pos <= 1.05 and 
                   angular_vel < 2.5)
            
        elif self.current_task == 4:  #POSIZIONE ACCOVACCIATA
            return (1.0 <= z_pos <= 1.25 and 
                   angular_vel < 2.0)
            
        elif self.current_task == 5:  #PREPARAZIONE ALZATA
            return (z_pos > task['success_threshold'] and 
                   angular_vel < 1.8)
            
        elif self.current_task == 6:  #ALZARSI IN PIEDI
            return (z_pos > task['success_threshold'] and 
                   angular_vel < 2.0)
            
        elif self.current_task == 7:  #TASK 8 - EQUILIBRIO STABILE
            #DEVE SUPERARE LA SOGLIA ED AVERE BUONA STABILITÀ
            basic_success = self.steps_standing >= self.dynamic_threshold
            
            if basic_success:
                #VERIFICA STABILITÀ AVANZATA
                stability_score = self._calculate_stability_score()
                is_stable_enough = stability_score > 0.7  
                
                #VERIFICA CHE NON STIA OSCIALLANDO TROPPO
                oscillation_ok = self._calculate_oscillation_penalty() < 2.0
                
                return is_stable_enough and oscillation_ok
            
            return False
        
        return False
    
    def _update_curriculum_progress(self):
        """AGGIORNAMENTO DEL PROGRESSO"""
        self.task_success_count += 1
        
        task = self.tasks[self.current_task]
        
        #STATISTICHE DI PROGRESSO
        success_rate = self.task_success_count / max(self.episode_count, 1)
        
        #CONTROLLA SE AVANZARE AL TASK SUCCESSIVO
        should_advance = (
            self.task_success_count >= task['success_target'] and
            self.episode_count >= task['min_episodes'] and
            self.current_task < len(self.tasks) - 1
        )
        
        #AVANZAMENTO FORZATO PER TIMEOUT
        force_advance = (
            self.episode_count >= task['max_episodes'] and
            self.current_task < len(self.tasks) - 1
        )
        
        if should_advance or force_advance:
            self._advance_to_next_task(force_advance)
        
        #GESTIONE SPECIALE PER ULTIMO TASK
        elif self.current_task == 7 and self.task_success_count >= task['success_target']:
            self._handle_final_task_completion()
    
    def _advance_to_next_task(self, forced=False):
        """AVANZA AL TASK SUCCESSIVO"""
        old_task = self.current_task
        self.current_task += 1
        old_count = self.task_success_count
        self.task_success_count = 0
        self.episode_count = 0
        
        #RESET DELLE METRICHE
        self.best_height_achieved = 0.0
        self.max_steps_standing = 0
        
        status = "FORZATO" if forced else "COMPLETATO"
        
        print(f"\n{status} - CURRICULUM AVANZATO!")
        print(f"   Da: Task {old_task + 1} - {self.tasks[old_task]['name']} ({old_count} successi)")
        print(f"   A:  Task {self.current_task + 1} - {self.tasks[self.current_task]['name']}")
        print(f"   Target: {self.tasks[self.current_task]['success_target']} successi")
        print(f"   Descrizione: {self.tasks[self.current_task]['description']}")
    
    def _handle_final_task_completion(self):
        """GESTISCE IL COMPLETAMENTO FINALE DEL TASK CON SOGLIA DINAMICA"""
        self.threshold_increases += 1
        #INCREMENTI PIÙ GRADUALI: 25 -> 30 -> 35 -> 40 -> 50 (MAX)
        increment = min(5 * self.threshold_increases, 25)
        self.dynamic_threshold = min(25 + increment, 50)
        self.task_success_count = 0  #Reset per nuova soglia
        
        print(f"\n TASK FINALE COMPLETATO!")
        print(f"   Nuova sfida: mantenere equilibrio per {self.dynamic_threshold} step")
        print(f"   Incremento #{self.threshold_increases} (+{increment} step)")
        
        # Aggiorna la soglia nel task
        self.tasks[7]['success_threshold'] = self.dynamic_threshold
    
    def get_detailed_curriculum_info(self):
        """Informazioni dettagliate sullo stato del curriculum"""
        task = self.tasks[self.current_task]
        return {
            'current_task': self.current_task + 1,
            'total_tasks': len(self.tasks),
            'task_name': task['name'],
            'task_description': task['description'],
            'success_count': self.task_success_count,
            'success_target': task['success_target'],
            'episode_count': self.episode_count,
            'min_episodes': task['min_episodes'],
            'max_episodes': task['max_episodes'],
            'success_rate': self.task_success_count / max(self.episode_count, 1),
            'best_height': self.best_height_achieved,
            'max_steps_standing': self.max_steps_standing,
            'current_threshold': self.dynamic_threshold if self.current_task == 7 else task['success_threshold']
        }
    
    def get_wrapper_attr(self, name):
        """Metodo per accedere agli attributi del wrapper dall'esterno"""
        if hasattr(self, name):
            return getattr(self, name)
        return None

#ALIAS per retrocompatibilità  
CurriculumWrapper = ImprovedCurriculumWrapper