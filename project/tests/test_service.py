from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT


class ServiceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_path = PROJECT_ROOT / "artifacts" / "model.joblib"
        metadata_path = PROJECT_ROOT / "artifacts" / "model_metadata.json"
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

    def test_predict_endpoint(self) -> None:
        payload = {
            "tasks": [
                {
                    "priority": "high",
                    "assignee_experience": 2.0,
                    "estimated_hours": 32.0,
                    "actual_progress": 30.0,
                    "days_since_created": 17,
                    "comments_count": 7,
                    "status": "blocked",
                    "team_size": 4,
                    "task_type": "bug",
                    "has_blockers": True,
                    "blockers_count": 2,
                    "recent_reassignments": 1,
                    "sprint_phase": "release_week",
                    "priority_score": 3,
                    "workload_ratio": 2.8,
                    "is_customer_facing": True,
                    "requires_review": True
                }
            ]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_name"], "logistic_regression")
        self.assertEqual(len(body["predictions"]), 1)
        self.assertIn("overdue_probability", body["predictions"][0])

    def test_invalid_priority_score_is_rejected(self) -> None:
        payload = {
            "tasks": [
                {
                    "priority": "low",
                    "assignee_experience": 5.0,
                    "estimated_hours": 8.0,
                    "actual_progress": 90.0,
                    "days_since_created": 3,
                    "comments_count": 1,
                    "status": "done",
                    "team_size": 4,
                    "task_type": "documentation",
                    "has_blockers": False,
                    "blockers_count": 0,
                    "recent_reassignments": 0,
                    "sprint_phase": "mid_sprint",
                    "priority_score": 3,
                    "workload_ratio": 0.5,
                    "is_customer_facing": False,
                    "requires_review": False
                }
            ]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
