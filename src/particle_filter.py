from predict import predict_next_log_with_constraints
from constraints_checker import check_constraints
from constraints import constraints
from activity import ActivityPrediction
from interactive_constraint_manager import InteractiveConstraintManager

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
        self.k = k  # Numero massimo di attività successive per ogni particella
        self.max_particles = max_particles  # Limite massimo di particelle attive
        self.particles = []
        self.constraint_manager = InteractiveConstraintManager()

    def initialize_particles(self, initial_activity):
        """ Inizializza solo una particella con l'attività iniziale """
        self.particles = [[ActivityPrediction(initial_activity, 1.0)]]
        print(f"  - Particella iniziale impostata: {initial_activity}")

    def step(self, step_num):
        """ Avanza di uno step, elaborando SOLO le particelle esistenti """
        print(f"\n=== STEP {step_num} ===")
        print(f"[INFO] Step {step_num}, particelle attive: {len(self.particles)}")

        new_particles = []
        for particle in self.particles:
            predicted_activities = predict_next_log_with_constraints(
                self.model, self.tokenizer, particle, self.label_map, self.device, num_candidates=self.k
            )

            if not predicted_activities:
                continue

            for activity in predicted_activities:
                new_particle = particle + [activity]
                new_particles.append(new_particle)

        self.particles = new_particles  # Aggiorniamo la lista di particelle
        print(f"[INFO] Particelle generate al termine dello step {step_num}: {len(self.particles)}")

    def run(self, steps):
        """ Esegue il filtro per un numero di step definito """
        for step_num in range(1, steps + 1):
            if not self.particles:
                print("[INFO] Nessuna particella rimanente. Fine del processo.")
                break

            self.step(step_num)  # Ora ogni step elabora solo le particelle esistenti

        return self.particles



    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()
