from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from bank_predictor.config import DATA_PATH, FEATURE_FIELDS
from bank_predictor.ml import load_dataset, load_metadata, load_pipeline, normalize_record
from train import main as train_main


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train_main()

        from app.main import app

        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        cls.pipeline = load_pipeline()
        cls.metadata = load_metadata()
        cls.dataset = load_dataset(DATA_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def build_sample_payload(self) -> dict[str, object]:
        row = self.dataset.iloc[0].to_dict()
        trimmed_row = {field: row[field] for field in FEATURE_FIELDS}
        return normalize_record(trimmed_row)

    def test_health_route(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["artifacts_ready"])

    def test_metadata_route(self) -> None:
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["feature_order"], list(FEATURE_FIELDS))
        self.assertEqual(payload["model_name"], "GradientBoostingClassifier")
        self.assertIn("decision_threshold", payload)
        self.assertIn("job", payload["categorical_options"])
        self.assertIn("accuracy", payload["metrics"])
        self.assertGreater(payload["metrics"]["f1"], 0.45)
        self.assertGreater(payload["metrics"]["recall"], 0.45)

    def test_predict_route_matches_pipeline(self) -> None:
        sample_payload = self.build_sample_payload()

        direct_probability_yes = float(self.pipeline.predict_proba(self.dataset.loc[[0], FEATURE_FIELDS])[0][1])
        expected_label = "yes" if direct_probability_yes >= self.metadata["decision_threshold"] else "no"
        response = self.client.post("/predict", json=sample_payload)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["label"], expected_label)
        self.assertAlmostEqual(payload["probability_yes"] + payload["probability_no"], 1.0, places=6)
        self.assertAlmostEqual(payload["probability_yes"], direct_probability_yes, places=6)
        self.assertAlmostEqual(payload["decision_threshold"], self.metadata["decision_threshold"], places=6)

    def test_predict_validation_error(self) -> None:
        response = self.client.post("/predict", json={"age": 30})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
