from Declare4Py.Utils.Declare.Checkers import ResponseTemplate, PrecedenceTemplate, ExistenceTemplate

# Esempio di constraints definiti come lista di dizionari
constraints = [
    {
        "template": ResponseTemplate(),  # Vincolo di tipo "Response"
        "activities": ["A", "B"],
        "condition": ["A", "B", 24]  # ad esempio, entro 24 ore
    },
    {
        "template": PrecedenceTemplate(),  # Vincolo di tipo "Precedence"
        "activities": ["A", "B"],
        "condition": ["A", "B", 24]
    },
    {
        "template": ExistenceTemplate(),  # Vincolo di tipo "Existence"
        "activities": ["C"],
        "condition": ["C", None, None]  # Non richiede correlazione o tempo
    }
]
