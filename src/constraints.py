import pandas as pd

class ConstraintTemplate:
    def __init__(self, name, supports_cardinality=False, is_binary=False):
        self.name = name
        self.supports_cardinality = supports_cardinality
        self.is_binary = is_binary

    def __repr__(self):
        return f"ConstraintTemplate({self.name})"

ResponseTemplate = ConstraintTemplate("Response", is_binary=True)
PrecedenceTemplate = ConstraintTemplate("Precedence", is_binary=True)
ExistenceTemplate = ConstraintTemplate("Existence", supports_cardinality=True)

def generate_dynamic_constraints(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    if not {"case", "activity", "timestamp"}.issubset(df.columns):
        raise ValueError("Il file CSV deve contenere le colonne 'case', 'activity' e 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    activity_counts = df["activity"].value_counts()
    most_frequent_activities = activity_counts.index[:5].tolist()  # Prendiamo le 5 attività più comuni

    first_activities = df.sort_values(by="timestamp").groupby("case").first()["activity"].value_counts()
    last_activities = df.sort_values(by="timestamp").groupby("case").last()["activity"].value_counts()

    most_common_start = first_activities.idxmax()
    most_common_end = last_activities.idxmax()

    constraints = []

    constraints.append({
        "template": PrecedenceTemplate,
        "activities": [most_common_start, most_frequent_activities[0]],
        "condition": [most_common_start, most_frequent_activities[0], None]
    })

    constraints.append({
        "template": ResponseTemplate,
        "activities": [most_common_start, most_common_end],
        "condition": [most_common_start, most_common_end, 24]
    })

    return constraints

csv_path = "/kaggle/working/ProgettoTirocinio/dataset/helpdesk.csv"
constraints = generate_dynamic_constraints(csv_path)
