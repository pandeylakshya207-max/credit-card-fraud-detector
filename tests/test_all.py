"""
test_all.py — Test suite for the Credit Card Fraud Detector.

Run:
    python3 tests/test_all.py
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "\033[92m✅\033[0m" if sys.stdout.isatty() else "PASS"
FAIL = "\033[91m❌\033[0m" if sys.stdout.isatty() else "FAIL"

results = []


def test(name):
    def decorator(fn):
        results.append((name, fn))
        return fn
    return decorator


@test("Model file loads")
def test_model_loads():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None
    assert hasattr(model, "predict_proba")


@test("Time scaler loads and has sane statistics")
def test_time_scaler():
    with open("time_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # Time in the dataset spans ~2 days (0 to ~172792 seconds)
    assert 50000 < scaler.mean_[0] < 150000, f"Time scaler mean looks wrong: {scaler.mean_[0]}"


@test("Amount scaler loads and has sane statistics")
def test_amount_scaler():
    with open("amount_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # Average transaction amount should be a plausible dollar figure, not seconds
    assert 1 < scaler.mean_[0] < 1000, f"Amount scaler mean looks wrong: {scaler.mean_[0]}"


@test("Time and Amount scalers are NOT the same fitted values (regression test for the shared-scaler bug)")
def test_scalers_are_different():
    with open("time_scaler.pkl", "rb") as f:
        time_scaler = pickle.load(f)
    with open("amount_scaler.pkl", "rb") as f:
        amount_scaler = pickle.load(f)
    assert abs(time_scaler.mean_[0] - amount_scaler.mean_[0]) > 1000, (
        "Time and Amount scalers have suspiciously similar means -- "
        "this is the exact bug where one shared scaler object got refit, "
        "silently overwriting the first column's fitted statistics."
    )


@test("Health endpoint reports model loaded")
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


@test("Predict endpoint returns valid probability for a known legit transaction")
def test_predict_legit():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "Time": 0, "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
        "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
        "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] is False, "This is a known real legit transaction from the dataset"


@test("Predict endpoint flags a known real fraud case at a lower threshold")
def test_predict_fraud():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    payload = {
        "Time": 406.0, "V1": -2.312226542, "V2": 1.951992011, "V3": -1.609850732,
        "V4": 3.997905588, "V5": -0.522187865, "V6": -1.426545319, "V7": -2.537387306,
        "V8": 1.391657248, "V9": -2.770089277, "V10": -2.772272145, "V11": 3.202033207,
        "V12": -2.899907388, "V13": -0.595221881, "V14": -4.289253782, "V15": 0.38972412,
        "V16": -1.14074718, "V17": -2.830055675, "V18": -0.016822468, "V19": 0.416955705,
        "V20": 0.126910559, "V21": 0.517232371, "V22": -0.035049369, "V23": -0.465211076,
        "V24": 0.320198199, "V25": 0.044519167, "V26": 0.177839798, "V27": 0.261145003,
        "V28": -0.143275875, "Amount": 0.0,
    }
    r = client.post("/predict?threshold=0.3", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["fraud_probability"] > 0.3, (
        f"Known fraud case scored too low: {body['fraud_probability']}"
    )
    assert body["is_fraud"] is True


@test("Predict endpoint rejects malformed input")
def test_predict_malformed():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/predict", json={"Time": 0, "Amount": 100})  # missing V1-V28
    assert r.status_code == 422, "Should reject incomplete transaction data with a validation error"


@test("Metrics file has plausible values")
def test_metrics_plausible():
    import json
    with open("metrics.json") as f:
        m = json.load(f)
    assert m["total_transactions"] == 284807
    assert m["total_fraud"] == 492
    assert 0 < m["approach_balanced"]["pr_auc"] < 1
    assert 0 < m["approach_smote"]["pr_auc"] < 1
    assert m["baseline_frauds_caught"] == 0, "Baseline should genuinely catch zero fraud"


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detector — Test Suite")
    print("=" * 60)
    passed, failed = 0, 0
    failures = []
    for name, fn in results:
        try:
            fn()
            print(f"  {PASS}  {name}")
            passed += 1
        except Exception as e:
            print(f"  {FAIL}  {name}")
            print(f"       {e}")
            failed += 1
            failures.append(name)
    print("=" * 60)
    print(f"  Results: {passed}/{len(results)} passed  |  {failed} failed")
    print("=" * 60)
    if failed:
        print("\nFAILED TESTS:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nALL TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
