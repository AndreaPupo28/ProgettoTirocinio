class ActivityPrediction:
    def __init__(self, name, probability):
        self.name = name  # Nome dell'attività
        self.probability = probability  # Probabilità associata

    def __repr__(self):
        return f"ActivityPrediction(name={self.name}, probability={self.probability:.4f})"
