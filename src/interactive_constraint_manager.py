from constraints import ConstraintTemplate

class InteractiveConstraintManager:
    def __init__(self, user_constraints=[]):
        self.user_constraints = user_constraints

    def request_constraints(self, current_length):
        # Usa solo i vincoli pre-caricati, niente input manuale
        print(f"\n[RICHIESTA VINCOLI] Lunghezza tracce attuali: {current_length}.")
        if self.user_constraints:
            print("Vincoli caricati automaticamente:")
            for constraint in self.user_constraints:
                print(constraint)
        else:
            print("Nessun vincolo specificato. Procedo senza nuovi vincoli.")

    def get_constraints(self):
        return self.user_constraints
