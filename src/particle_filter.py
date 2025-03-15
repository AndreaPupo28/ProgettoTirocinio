from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager

class ParticleFilter:
    def __init__(self, model, tokenizer, label_map, device, k=3, max_particles=100):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.k = k  # Numero di attività più probabili da selezionare a ogni step
        self.max_particles = max_particles  # Limite massimo di particelle attive
        self.particles = []
        self.constraint_manager = InteractiveConstraintManager()

    def initialize_particles(self, initial_activity):
        self.particles = [[ActivityPrediction(initial_activity, 1.0)]]
        print(f"  - Particella iniziale: {initial_activity}")

    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()

    def step(self):
        new_particles = []
        print(f"\n[INFO] Step attuale, particelle attive: {len(self.particles)}")

        for particle in self.particles:
            if len(new_particles) >= self.max_particles:
                break  # Evitiamo la crescita esponenziale

            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=self.k
            )

            if not predicted_sequences or not predicted_sequences[0]:
                continue

            # Selezioniamo le k attività più probabili per ogni sequenza
            for predicted_name, predicted_prob in sorted(predicted_sequences[0], key=lambda x: x[1], reverse=True)[:self.k]:
                if len(new_particles) >= self.max_particles:
                    break

                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                sequence_names = [act.name for act in new_particle]

                # Evitiamo cicli ripetitivi nelle ultime 5 attività
                if len(sequence_names) > 5 and len(sequence_names[-5:]) != len(set(sequence_names[-5:])):
                    continue

                current_constraints = self.sense_environment(new_particle)
                if check_constraints(" ".join(sequence_names), current_constraints, detailed=False, completed=True):
                    new_particles.append(new_particle)

        print(f"[INFO] Nuove particelle generate: {len(new_particles)}")
        self.particles = new_particles

    def run(self, steps):
        for step_num in range(steps):
            print(f"\n=== STEP {step_num + 1}/{steps} ===")
            self.step()

            if not self.particles:
                print("[INFO] Nessuna particella rimanente. Fine del processo.")
                break

        return self.particles
