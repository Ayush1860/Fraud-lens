"""Synthetic load-test traffic: valid requests, a fixed card pool, time moving forward."""

from fraudlens.service.schemas import ScoreRequest
from fraudlens.tools.traffic import SyntheticTraffic


def test_requests_are_valid_unique_and_move_forward_in_time() -> None:
    traffic = SyntheticTraffic(n_cards=50, seed=1)

    bodies = [traffic.next() for _ in range(500)]

    requests = [ScoreRequest.model_validate(body) for body in bodies]
    assert len({r.transaction_id for r in requests}) == 500
    times = [r.event_time for r in requests]
    assert times == sorted(times)
    assert len({r.card_id for r in requests}) <= 50


def test_cards_repeat_so_history_builds_up() -> None:
    traffic = SyntheticTraffic(n_cards=20, seed=2)

    cards = [traffic.next()["card_id"] for _ in range(200)]

    assert max(cards.count(card) for card in set(cards)) >= 5


def test_two_generators_with_one_seed_agree_but_ids_differ_by_run() -> None:
    a, b = (
        SyntheticTraffic(n_cards=10, seed=3, run="x"),
        SyntheticTraffic(n_cards=10, seed=3, run="y"),
    )

    first_a, first_b = a.next(), b.next()

    assert first_a["amount"] == first_b["amount"]
    assert first_a["transaction_id"] != first_b["transaction_id"]
