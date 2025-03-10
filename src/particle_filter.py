import torch
from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager
import random

# Variabile globale per gestire la terminazione del processo
process_terminated = False

class ParticleFilter:
    def __init__(self, model, tokenizer, label_map, device, num_particles=50):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.num_particles = num_particles
        self.particles = []
        self.constraint_manager = InteractiveConstraintManager()

    def step(self):
        global process_terminated
        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0

        self.constraint_manager.request_constraints(current_length)

        MAX_PARTICLES = 10000

        for particle in self.particles:
            if len(new_particles) >= MAX_PARTICLES:
                print("Raggiunto il limite massimo di particelle. Interruzione della generazione.")
                break

            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=2
            )

            if not predicted_sequences or not predicted_sequences[0]:
                print(f"Fine della traccia per la particella: {[act.name for act in particle]} - nessuna nuova attività da predire.")
                continue

            for predicted_name, predicted_prob in predicted_sequences[0]:
                if len(new_particles) >= MAX_PARTICLES:
                    break
                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                current_constraints = self.sense_environment(new_particle)
                if check_constraints(" ".join([act.name for act in new_particle]), current_constraints, detailed=False, completed=True):
                    new_particles.append(new_particle)
                    print(f"Prossima attività predetta: {predicted_name} con probabilità {predicted_prob:.4f}\n")

        if not new_particles:
            print("Nessuna nuova particella generata. Terminazione del processo.")
            process_terminated = True

        self.particles = new_particles

    def run(self, steps):
        for step_num in range(steps):
            print(f"\n=== STEP {step_num + 1}/{steps} ===")
            self.step()
            if not self.particles:
                print("Nessuna particella rimanente. Fine del processo.")
                break
        return self.particles
