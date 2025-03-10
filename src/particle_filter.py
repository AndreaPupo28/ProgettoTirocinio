import torch
from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager
import random

# Variabile globale per segnalare la terminazione del processo
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

    def initialize_particles(self, initial_activities):
        self.particles = [[ActivityPrediction(activity, 1.0)] for activity in initial_activities]
        print("\n--------------------------------------")
        print("Inizio della generazione delle particelle iniziali")
        print("--------------------------------------\n")
        for particle in self.particles:
            print(f"Particella iniziale: {[act.name for act in particle]}")

    def sense_environment(self, particles):
        current_constraints = constraints + self.constraint_manager.get_constraints()
        print("Vincoli usati:", current_constraints)
        return current_constraints

    def step(self):
        global process_terminated
        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0

        self.constraint_manager.request_constraints(current_length)

        MAX_PARTICLES = 10000

        for particle in self.particles:
            input_text = "<SOS> " + " ".join([act.name for act in particle]) + " <EOS>"
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=5
            )

            if not predicted_sequences or not predicted_sequences[0]:
                print(f"Fine della traccia per la particella: {[act.name for act in particle]} - nessuna nuova attività da predire.")
                continue

            for predicted_name, predicted_prob in predicted_sequences[0]:
                if len(new_particles) >= MAX_PARTICLES:
                    process_terminated = True
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
        global process_terminated
        for step_num in range(steps):
            print(f"\n=== STEP {step_num + 1}/{steps} ===")
            self.step()
            if not self.particles or process_terminated:
                print("Nessuna particella rimanente o processo terminato. Fine del processo.")
                break
        return self.particles
