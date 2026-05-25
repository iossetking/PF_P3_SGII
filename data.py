import numpy as np
import pandas as pd

DATASET_PATH = "clean_gym_data.csv"

NUMERIC_FEATURES = [
    "gdp_per_capita_usd",
    "urban_population_percentage",
    "fitness_participation_rate",
    "obesity_rate",
    "average_membership_cost_usd",
    "insufficient_physical_activity_pct",
    "year",
]
TARGET = "gym_penetration_rate"


def load_raw(dataset_path=DATASET_PATH):
    return pd.read_csv(dataset_path)


def load_data(dataset_path=DATASET_PATH):
    df = load_raw(dataset_path)

    region_dummies = pd.get_dummies(df["region"], prefix="region").astype(np.float32)
    region_categories = np.array(sorted(df["region"].unique()))

    X_numeric = df[NUMERIC_FEATURES].astype(np.float32).to_numpy()
    X_region = region_dummies.to_numpy()
    X = np.concatenate([X_numeric, X_region], axis=1)

    # log1p reduces right-skew
    y = np.log1p(df[TARGET].astype(np.float32).to_numpy())

    return X, y, region_categories


def sample_rows(n=5, seed=42, dataset_path=DATASET_PATH):
    df = load_raw(dataset_path)
    sample = df.sample(n=n, random_state=seed)

    rows = []
    for _, row in sample.iterrows():
        features: dict[str, float | str] = {f: float(str(row[f])) for f in NUMERIC_FEATURES}
        features["region"] = str(row["region"])
        rows.append((str(row["country"]), int(str(row["year"])), float(str(row[TARGET])), features))

    return rows
