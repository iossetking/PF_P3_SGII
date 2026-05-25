import data
import model
import threads

MODEL_PATH = "gym_model.keras"
SCALER_PATH = "gym_scaler.pkl"


def prediction_interface():
    print("\nGym Penetration Rate by Country Predictions")
    print("-" * 60)

    samples = data.sample_rows(n=8, seed=0)
    tasks = [(f"{country}|{year}", features) for country, year, _, features in samples]
    result_map = dict(threads.run_predictions(tasks, model_path=MODEL_PATH, scaler_path=SCALER_PATH))

    for country, year, actual, _ in samples:
        predicted = result_map[f"{country}|{year}"]
        error = predicted - actual
        print(
            f"{country:<25} ({year}) | "
            f"actual: {actual:.4f} | "
            f"predicted: {predicted:.4f} | "
            f"error: {error:+.4f}"
        )


def future_prediction_interface():
    print("\nFuture Gym Penetration Rate Predictions (2030)")
    print("-" * 60)

    scenarios = [
        ("Germany (optimistic)", {
            "gdp_per_capita_usd": 58000.0,
            "urban_population_percentage": 0.78,
            "fitness_participation_rate": 0.50,
            "obesity_rate": 0.20,
            "average_membership_cost_usd": 42.0,
            "insufficient_physical_activity_pct": 0.22,
            "year": 2030.0,
            "region": "Western Europe",
        }),
        ("Brazil (moderate growth)", {
            "gdp_per_capita_usd": 12000.0,
            "urban_population_percentage": 0.90,
            "fitness_participation_rate": 0.35,
            "obesity_rate": 0.26,
            "average_membership_cost_usd": 28.0,
            "insufficient_physical_activity_pct": 0.38,
            "year": 2030.0,
            "region": "South America",
        }),
        ("Kenya (emerging market)", {
            "gdp_per_capita_usd": 2500.0,
            "urban_population_percentage": 0.35,
            "fitness_participation_rate": 0.25,
            "obesity_rate": 0.10,
            "average_membership_cost_usd": 15.0,
            "insufficient_physical_activity_pct": 0.18,
            "year": 2030.0,
            "region": "Sub-Saharan Africa",
        }),
    ]

    results = threads.run_predictions(scenarios, model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    for label, predicted in results:
        print(f"{label:<35} | predicted: {predicted:.4f}")


def main():
    loaded = model.load_model_and_meta(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    if loaded is None:
        print("No trained model found. Training new model...")
        model.train_and_save(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    else:
        print("Model and metadata loaded successfully.")

    prediction_interface()
    future_prediction_interface()


if __name__ == "__main__":
    main()
