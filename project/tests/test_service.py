from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT


class ServiceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_path = PROJECT_ROOT / "models" / "model.joblib"
        metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise unittest.SkipTest("Model artifacts are missing. Run `python -m src.train` before tests.")

        from src.service import app

        cls.test_client = TestClient(app)
        cls.client = cls.test_client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "test_client"):
            cls.test_client.__exit__(None, None, None)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["model_loaded"])

    def test_predict_endpoint(self) -> None:
        payload = {
            "issues": [
                {
                    "issue_type": "Bug",
                    "priority": "High",
                    "has_priority": True,
                    "component_present": True,
                    "summary_length": 92,
                    "summary_word_count": 12,
                    "description_length": 1840,
                    "description_word_count": 245
                }
            ]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_version"], "v1")
        self.assertEqual(len(body["predictions"]), 1)
        self.assertIn("prediction", body["predictions"][0])
        self.assertIn("probability", body["predictions"][0])
        self.assertIn("delay_probability", body["predictions"][0])

    def test_invalid_summary_length_is_rejected(self) -> None:
        payload = {
            "issues": [
                {
                    "issue_type": "Bug",
                    "priority": "Low",
                    "has_priority": True,
                    "component_present": True,
                    "summary_length": -1,
                    "summary_word_count": 5,
                    "description_length": 200,
                    "description_word_count": 30
                }
            ]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
