from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager

class ParticleFilter:
    def __init__(self, model, tokenizer, label_map, device, num_particles=50, max_particles=100):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.num_particles = num_particles
        self.max_particles = max_particles  # Limite massimo di particelle attive
        self.particles = []
        self.constraint_manager = InteractiveConstraintManager()

    def initialize_particles(self, initial_activities):
        self.particles = [[ActivityPrediction(activity, 1.0)] for activity in initial_activities]
        print(f"\n[INFO] Generazione di {len(self.particles)} particelle iniziali:")
        for particle in self.particles:
            print(f"  - Particella iniziale: {[act.name for act in particle]}")

    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()

    def step(self):
        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0
        self.constraint_manager.request_constraints(current_length)
        
        print(f"[INFO] Step attuale: {current_length}, particelle attive: {len(self.particles)}")

        for particle in self.particles:
            if len(new_particles) >= self.max_particles:
                break  # Evitiamo la crescita esponenziale
            
            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=3
            )  # Ridotto a 3 candidati
            
            if not predicted_sequences or not predicted_sequences[0]:
                continue

            for predicted_name, predicted_prob in sorted(predicted_sequences[0], key=lambda x: x[1], reverse=True)[:2]:
                if len(new_particles) >= self.max_particles:
                    break
                
                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                sequence_names = [act.name for act in new_particle]
                
                # Controllo per evitare cicli ripetitivi negli ultimi 5 passi
                if len(sequence_names) > 5 and len(sequence_names[-5:]) != len(set(sequence_names[-5:])):
                    continue
                
                current_constraints = self.sense_environment(new_particle)
                if check_constraints(" ".join(sequence_names), current_constraints, detailed=False, completed=True):
                    new_particles.append(new_particle)
        
        print(f"[INFO] Nuove particelle generate: {len(new_particles)}")
        self.particles = new_particles

    def run(self, steps):
        stagnation_counter = 0
        prev_particle_count = len(self.particles)

        for step_num in range(steps):
            print(f"\n=== STEP {step_num + 1}/{steps} ===")
            self.step()
            
            if not self.particles:
                print("[INFO] Nessuna particella rimanente. Fine del processo.")
                break

            # Controllo stagnazione
            if len(self.particles) == prev_particle_count:
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # Reset se cambia qualcosa
            prev_particle_count = len(self.particles)

            if stagnation_counter >= 3:
                print("[INFO] Nessuna evoluzione nelle particelle, interruzione anticipata.")
                break
        
        return self.particles
