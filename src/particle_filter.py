import torch
from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager

class ParticleFilter:
    def __init__(self, model, tokenizer, label_map, device, num_particles=3, constraints=[]):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.num_particles = num_particles
        self.particles = []
        # Passa direttamente i vincoli caricati dal file JSON
        self.constraint_manager = InteractiveConstraintManager(user_constraints=constraints)

    def initialize_particles(self, initial_activities):
        self.particles = [[ActivityPrediction(activity, 1.0)] for activity in initial_activities]

    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()

    def step(self):
        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0

        # Richiede i vincoli senza input manuale
        self.constraint_manager.request_constraints(current_length)

        for particle in self.particles[:3]:  # Limita a 5 particelle per step
            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=2
            )
            if not predicted_sequences or not predicted_sequences[0]:
                continue

            for predicted_name, predicted_prob in predicted_sequences[0]:
                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                current_constraints = self.sense_environment(new_particle)
                if check_constraints(" ".join([act.name for act in new_particle]), current_constraints, detailed=False, completed=True):
                    new_particles.append(new_particle)

            if len(new_particles) >= 50:  # Limita il numero di nuove particelle generate
                break

        self.particles = new_particles

    def run(self, steps=1):
        for step_num in range(steps):
            print(f"STEP {step_num}: {len(self.particles)} particelle attive")
            self.step()
            if not self.particles:
                print("Nessuna particella rimanente. Fine del processo.")
                break
        return self.particles
