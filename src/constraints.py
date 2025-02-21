# constraints.py

# Simuliamo la struttura che `constraints_checker` si aspetta
class ConstraintTemplate:
    """ Classe generica per simulare i template di vincoli senza importarli. """
    def __init__(self, name, supports_cardinality=False, is_binary=False):
        self.name = name
        self.supports_cardinality = supports_cardinality
        self.is_binary = is_binary

    def __repr__(self):
        return f"ConstraintTemplate({self.name})"

# Creiamo i template compatibili con `constraints_checker.py`
ResponseTemplate = ConstraintTemplate("Response", is_binary=True)
PrecedenceTemplate = ConstraintTemplate("Precedence", is_binary=True)
ExistenceTemplate = ConstraintTemplate("Existence", supports_cardinality=True)

# Definizione dei constraints utilizzando i template simulati
constraints = [
    {
        "template": ResponseTemplate,  # Vincolo di tipo "Response"
        "activities": ["A", "B"],
        "condition": ["A", "B", 24],  # Ad esempio, entro 24 ore
        "n": 1  # Se richiesto da qualche vincolo
    },
    {
        "template": PrecedenceTemplate,  # Vincolo di tipo "Precedence"
        "activities": ["A", "B"],
        "condition": ["A", "B", 24]
    },
    {
        "template": ExistenceTemplate,  # Vincolo di tipo "Existence"
        "activities": ["C"],
        "condition": ["C", None, None],
        "n": 1  # Necessario perché il template supporta cardinalità
    }
]

