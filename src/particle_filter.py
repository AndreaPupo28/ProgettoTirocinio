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
        self.finished_particles = []  # Nuovo: particelle che hanno terminato (con "END OF SEQUENCE")
        self.constraint_manager = InteractiveConstraintManager()

    def initialize_particles(self, initial_activity):
        """ Inizializza solo una particella con l'attività iniziale """
        self.particles = [[ActivityPrediction(initial_activity, 1.0)]]
        print(f"  - Particella iniziale impostata: {initial_activity}")

    def step(self, step_num):
        """
        Avanza di uno step: per ogni particella esistente, vengono richieste le previsioni e si generano
        nuove particelle aggiungendo ESATTAMENTE k previsioni (o meno se non disponibili).
        """
        print(f"\n=== STEP {step_num} ===")
        num_particles_in = len(self.particles)
        print(f"[INFO] Step {step_num}, particelle attive: {num_particles_in}")

        new_particles = []
        current_length = len(self.particles[0]) if self.particles else 0
        print(f"\n Le tracce attuali hanno lunghezza {current_length}.")

        if step_num == 1:
            self.constraint_manager.request_constraints(current_length)

        for particle in self.particles:
            predicted_activities = predict_next_log_with_constraints(
                self.model, self.tokenizer, particle, self.label_map, self.device,
                num_candidates=self.k, constraint_manager=self.constraint_manager
            )

            if len(predicted_activities) < self.k:
                print(f"[WARNING] Particle {particle} restituisce solo {len(predicted_activities)} predizioni (minimo richiesto: {self.k}).")

            for predicted_activity in predicted_activities[:self.k]:
                new_particle = particle + [predicted_activity]
                # Ottieni i vincoli attuali per la particella
                current_constraints = self.sense_environment(new_particle)
                if check_constraints(
                        " ".join([act.name for act in new_particle]),
                        current_constraints,
                        detailed=False,
                        completed=True
                ):
                    if predicted_activity.name == "END OF SEQUENCE":
                        # La particella ha terminato la generazione
                        self.finished_particles.append(new_particle)
                        print(f"[INFO] Particella conclusa: {[act.name for act in new_particle]}")
                    else:
                        new_particles.append(new_particle)

        expected = num_particles_in * self.k
        print(f"[INFO] Particelle generate al termine dello step {step_num}: {len(new_particles)} (attese: {expected})")
        self.particles = new_particles

    def run(self, steps):
        """ Esegue il filtro per un numero di step definito """
        for step_num in range(1, steps + 1):
            if not self.particles:
                print("[INFO] Nessuna particella rimanente. Fine del processo.")
                break
            self.step(step_num)
        # Restituisce l'insieme complessivo: particelle concluse + quelle ancora in elaborazione
        all_particles = self.finished_particles + self.particles
        return all_particles

    def sense_environment(self, particles):
        return constraints + self.constraint_manager.get_constraints()

