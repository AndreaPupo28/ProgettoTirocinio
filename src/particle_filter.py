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
            input_text = " ".join([act.name for act in particle])
            predicted_sequences = predict_next_log_with_constraints(
                self.model, self.tokenizer, input_text, self.label_map, self.device, num_candidates=self.k
            )

            if not predicted_sequences or not predicted_sequences[0]:
                continue

            # Prendiamo al massimo `k` attività più probabili
            valid_candidates = sorted(predicted_sequences[0], key=lambda x: x[1], reverse=True)[:self.k]

            for predicted_name, predicted_prob in valid_candidates:
                new_particle = particle + [ActivityPrediction(predicted_name, predicted_prob)]
                new_particles.append(new_particle)

        self.particles = new_particles  # Sostituiamo le particelle con le nuove per il prossimo step
        print(f"[INFO] Particelle generate al termine dello step {step_num}: {len(self.particles)}")

    def run(self, steps):
        """ Esegue il filtro per un numero di step definito """
        for step_num in range(1, steps + 1):
            if not self.particles:
                print("[INFO] Nessuna particella rimanente. Fine del processo.")
                break

            self.step(step_num)  # Ora ogni step elabora solo le particelle esistenti e non genera tutte le sequenze in una volta

        return self.particles


    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()
