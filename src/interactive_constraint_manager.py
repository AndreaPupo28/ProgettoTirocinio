# class InteractiveConstraintManager:
#     def __init__(self, user_constraints=[]):
#         self.user_constraints = user_constraints
#
#     def request_constraints(self, current_length):
#         # Usa solo i vincoli pre-caricati, niente input manuale
#         print(f"\n[RICHIESTA VINCOLI] Lunghezza tracce attuali: {current_length}.")
#         if self.user_constraints:
#             print("Vincoli caricati automaticamente:")
#             for constraint in self.user_constraints:
#                 print(constraint)
#         else:
#             print("Nessun vincolo specificato. Procedo senza nuovi vincoli.")
#
#     def get_constraints(self):
#         return self.user_constraints

from constraints import ConstraintTemplate

class InteractiveConstraintManager:
    def __init__(self):
        self.user_constraints = []

    def request_constraints(self, current_length):
        #print(f"\n[RICHIESTA VINCOLI] Le tracce attuali hanno lunghezza {current_length}.")
        add_constraints = input("Vuoi aggiungere nuovi vincoli? (s/n): ").lower()

        if add_constraints == "s":
            while True:
                template_name = input("Inserisci il tipo di vincolo (es. 'Precedence', 'Response'): ")
                first_activity = input("Inserisci la prima attività: ")
                second_activity = input("Inserisci la seconda attività: ")
                time_constraint = input("Inserisci il vincolo temporale (in ore, o lascia vuoto se non necessario): ")

                try:
                    time_constraint = int(time_constraint) if time_constraint else None
                except ValueError:
                    print("Il vincolo temporale deve essere un numero intero. Riprova.")
                    continue

                constraint_template = ConstraintTemplate(
                    template_name, supports_cardinality=False, is_binary=True
                )
                constraint = {
                    "template": constraint_template,
                    "activities": [first_activity, second_activity],
                    "condition": [first_activity, second_activity, time_constraint]
                }
                self.user_constraints.append(constraint)
                print(f"Vincolo aggiunto: {constraint}")

                more_constraints = input("Vuoi aggiungere un altro vincolo? (s/n): ").lower()
                if more_constraints != "s":
                    break

    def get_constraints(self):
        return self.user_constraints
