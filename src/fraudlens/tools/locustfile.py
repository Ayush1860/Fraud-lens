"""Locust user for the scoring endpoint. Run through ``python -m fraudlens.tools.loadtest``.

Each simulated client sends ``FRAUDLENS_LOAD_PER_USER_RPS`` requests per second, so the
driver sets the number of users to reach a target rate. Requests come from one shared
:class:`~fraudlens.tools.traffic.SyntheticTraffic`, so card history builds up across users.
"""

import os

from locust import constant_throughput, task
from locust.contrib.fasthttp import FastHttpUser

from fraudlens.tools.traffic import SyntheticTraffic

TRAFFIC = SyntheticTraffic(
    n_cards=int(os.environ.get("FRAUDLENS_LOAD_CARDS", "5000")),
    run=os.environ.get("FRAUDLENS_LOAD_RUN", "load"),
)


class Scorer(FastHttpUser):
    """Posts one transaction at a time to the scoring endpoint."""

    wait_time = constant_throughput(float(os.environ.get("FRAUDLENS_LOAD_PER_USER_RPS", "5")))

    @task
    def score(self) -> None:
        """Score the next synthetic transaction."""
        self.client.post(
            "/v1/transactions/score",
            json=TRAFFIC.next(),
            headers={"X-API-Key": os.environ["FRAUDLENS_API_KEY"]},
            name="/v1/transactions/score",
        )
