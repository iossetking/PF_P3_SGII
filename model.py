import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data import DATASET_PATH, NUMERIC_FEATURES, load_data

MODEL_PATH = "gym_model.keras"
SCALER_PATH = "gym_scaler.pkl"


def build_model(input_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="huber",
        metrics=["mae"],
    )
    return model


def train_model(X, y, epochs=100, batch_size=32):
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Normalize features to [0, 1] range using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_model(input_dim=X_train.shape[1])

    # Train the model and track validation performance
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"\nValidation — loss (Huber): {val_loss:.4f}, MAE: {val_mae:.4f}")

    return model, scaler, history


def save_model(model, scaler, region_categories, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model.save(model_path)
    joblib.dump({"scaler": scaler, "region_categories": region_categories}, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def load_model_and_meta(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
    model = tf.keras.models.load_model(model_path)
    meta = joblib.load(scaler_path)
    scaler = meta["scaler"]
    region_categories = meta["region_categories"]
    return model, scaler, region_categories


def predict(features: dict, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    loaded = load_model_and_meta(model_path, scaler_path)
    if loaded is None:
        raise FileNotFoundError("No trained model found. Run train_and_save() first.")

    model, scaler, region_categories = loaded

    numeric = np.array([features[f] for f in NUMERIC_FEATURES], dtype=np.float32)

    region_enc = np.zeros(len(region_categories), dtype=np.float32)
    region = features.get("region", "")
    matches = np.where(region_categories == region)[0]
    if len(matches):
        region_enc[matches[0]] = 1.0

    x = np.concatenate([numeric, region_enc]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    log_pred = float(model.predict(x_scaled, verbose=0)[0][0])
    return float(np.expm1(log_pred))


def predict_with_loaded(loaded: tuple, features: dict) -> float:
    model, scaler, region_categories = loaded
    numeric = np.array([features[f] for f in NUMERIC_FEATURES], dtype=np.float32)
    region_enc = np.zeros(len(region_categories), dtype=np.float32)
    region = features.get("region", "")
    matches = np.where(region_categories == region)[0]
    if len(matches):
        region_enc[matches[0]] = 1.0
    x = np.concatenate([numeric, region_enc]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    log_pred = float(model(x_scaled, training=False)[0][0])
    return float(np.expm1(log_pred))


def predict_batch(feature_list: list[dict], model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    loaded = load_model_and_meta(model_path, scaler_path)
    if loaded is None:
        raise FileNotFoundError("No trained model found. Run train_and_save() first.")

    model, scaler, region_categories = loaded

    inputs = []
    for features in feature_list:
        numeric = np.array([features[f] for f in NUMERIC_FEATURES], dtype=np.float32)
        region_enc = np.zeros(len(region_categories), dtype=np.float32)
        region = features.get("region", "")
        matches = np.where(region_categories == region)[0]
        if len(matches):
            region_enc[matches[0]] = 1.0
        inputs.append(np.concatenate([numeric, region_enc]))

    X = np.array(inputs)
    X_scaled = scaler.transform(X)
    log_preds = model.predict(X_scaled, verbose=0)
    return [float(np.expm1(p[0])) for p in log_preds]


def train_and_save(dataset_path=DATASET_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    X, y, region_categories = load_data(dataset_path)
    model, scaler, _ = train_model(X, y)
    save_model(model, scaler, region_categories, model_path, scaler_path)


if __name__ == "__main__":
    train_and_save()
