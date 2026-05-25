import threading

import model as ml

MODEL_PATH = "gym_model.keras"
SCALER_PATH = "gym_scaler.pkl"


def run_predictions(
    tasks: list[tuple[str, dict]],
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    max_workers: int = 4,
) -> list[tuple[str, float]]:
    loaded = ml.load_model_and_meta(model_path, scaler_path)
    if loaded is None:
        raise FileNotFoundError("No trained model found. Run train_and_save() first.")

    results: dict[str, float] = {}
    lock = threading.Lock()
    semaphore = threading.Semaphore(max_workers)

    def worker(label: str, features: dict) -> None:
        with semaphore:
            predicted = ml.predict_with_loaded(loaded, features)
        with lock:
            results[label] = predicted

    thread_list = [
        threading.Thread(target=worker, args=(label, features))
        for label, features in tasks
    ]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    return [(label, results[label]) for label, _ in tasks]
