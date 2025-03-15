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
        """
        Avanza di uno step:
        Per ogni particella esistente, vengono richieste le previsioni e si generano
        nuove particelle aggiungendo ESATTAMENTE k previsioni (o meno se non disponibili).
        """
        print(f"\n=== STEP {step_num} ===")
        num_particles_in = len(self.particles)
        print(f"[INFO] Step {step_num}, particelle attive: {num_particles_in}")

        new_particles = []
        for particle in self.particles:
            predicted_activities = predict_next_log_with_constraints(
                self.model, self.tokenizer, particle, self.label_map, self.device, num_candidates=self.k
            )

            # Se per una particella non sono disponibili k candidati, viene generato un avviso
            if len(predicted_activities) < self.k:
                print(
                    f"[WARNING] Particle {particle} restituisce solo {len(predicted_activities)} predizioni (minimo richiesto: {self.k}).")

            # Prendiamo esattamente self.k candidati (se disponibili)
            for activity in predicted_activities[:self.k]:
                new_particles.append(particle + [activity])

        expected = num_particles_in * self.k
        print(f"[INFO] Particelle generate al termine dello step {step_num}: {len(new_particles)} (attese: {expected})")
        self.particles = new_particles

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
