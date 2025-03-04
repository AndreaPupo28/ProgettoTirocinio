import torch
from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager

class ParticleFilter:
    def __init__(self, model, tokenizer, label_map, device, num_particles=5):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.num_particles = num_particles
        self.particles = []
        self.constraint_manager = InteractiveConstraintManager()

    def initialize_particles(self, initial_activities):
        self.particles = [[ActivityPrediction(activity, 1.0)] for activity in initial_activities]
        print("\n--------------------------------------")
        print("Inizio della generazione delle particelle iniziali")
        print("--------------------------------------\n")
        for particle in self.particles:
            print(f"Particella iniziale: {[act.name for act in particle]}")

    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()

    def step(self):
        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0

        # Richiedi vincoli all'utente se la lunghezza delle particelle è x
        self.constraint_manager.request_constraints(current_length)

        for particle in self.particles:
            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=5
            )
            if not predicted_sequences or not predicted_sequences[0]:
                print(f"Fine della traccia per la particella: {[act.name for act in particle]} - nessuna nuova attività da predire.")
                continue

            for predicted_name, predicted_prob in predicted_sequences[0]:
                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                current_constraints = self.sense_environment(new_particle)
                if check_constraints(" ".join([act.name for act in new_particle]), current_constraints, detailed=False, completed=True):
                    new_particles.append(new_particle)
                    print(f"Prossima attività predetta: {predicted_name} con probabilità {predicted_prob:.4f}\n")

        self.particles = new_particles

    def run(self, steps=10):
        for step_num in range(steps):
            print(f"\n=== STEP {step_num + 1}/{steps} ===")
            self.step()
            if not self.particles:
                print("Nessuna particella rimanente. Fine del processo.")
                break
        return self.particles

