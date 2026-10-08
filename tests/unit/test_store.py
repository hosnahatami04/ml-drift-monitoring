import threading
from pathlib import Path

import pytest

from driftmon.monitoring.alerts import Alert
from driftmon.monitoring.windowing import is_window_complete, window_id_for
from driftmon.service.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db", window_size=5)


class TestWindowing:
    def test_window_ids_follow_arrival_order(self) -> None:
        assert window_id_for(1, 5) == 0
        assert window_id_for(5, 5) == 0
        assert window_id_for(6, 5) == 1
        assert window_id_for(11, 5) == 2

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            window_id_for(0, 5)
        with pytest.raises(ValueError):
            window_id_for(1, 0)

    def test_completion_rule(self) -> None:
        assert not is_window_complete(4, 5)
        assert is_window_complete(5, 5)


def test_prediction_roundtrip(store: Store) -> None:
    window_id, row_count = store.insert_prediction("req-1", {"age": 40}, 0.7)

    assert window_id == 0
    assert row_count == 1
    rows = store.window_predictions(0)
    assert len(rows) == 1
    assert rows[0].request_id == "req-1"
    assert rows[0].features == {"age": 40}
    assert rows[0].probability == 0.7
    assert rows[0].label is None


def test_label_attachment(store: Store) -> None:
    store.insert_prediction("req-1", {"age": 40}, 0.7)

    assert store.attach_label("req-1", 1)
    assert not store.attach_label("missing", 0)
    assert store.window_predictions(0)[0].label == 1


def test_complete_windows_are_reported_once(store: Store) -> None:
    for i in range(5):
        store.insert_prediction(f"req-{i}", {"age": 40}, 0.5)

    assert store.unprocessed_complete_windows() == [0]
    store.save_window_report(
        {
            "window_id": 0,
            "n_rows": 5,
            "results": [],
            "flagged_features": [],
            "feature_drift": False,
            "prediction_drift": False,
        }
    )
    assert store.unprocessed_complete_windows() == []
    assert store.processed_window_count() == 1


def test_performance_queue_requires_all_labels(store: Store) -> None:
    for i in range(5):
        store.insert_prediction(f"req-{i}", {"age": 40}, 0.5)
    store.save_window_report(
        {
            "window_id": 0,
            "n_rows": 5,
            "results": [],
            "flagged_features": [],
            "feature_drift": False,
            "prediction_drift": False,
        }
    )
    for i in range(4):
        store.attach_label(f"req-{i}", 1)
    assert store.windows_awaiting_performance() == []

    store.attach_label("req-4", 0)
    assert store.windows_awaiting_performance() == [0]

    store.mark_performance_checked(0, error_rate=0.2)
    assert store.windows_awaiting_performance() == []


def test_concurrent_writes_in_wal_mode(tmp_path: Path) -> None:
    store = Store(tmp_path / "concurrent.db", window_size=1000)
    errors: list[Exception] = []

    def writer(offset: int) -> None:
        try:
            for i in range(50):
                store.insert_prediction(f"req-{offset}-{i}", {"age": 40}, 0.5)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.prediction_count() == 200


def test_alert_roundtrip(store: Store) -> None:
    alert = Alert(
        window_id=3,
        type="feature_drift",
        severity="warning",
        features=("age",),
        details={"statistics": {}},
        created_at="2026-10-08T12:00:00+00:00",
    )

    store.insert_alert(alert)

    records = store.list_alerts()
    assert len(records) == 1
    assert records[0]["window_id"] == 3
    assert records[0]["features"] == ["age"]


def test_store_reopens_without_remigrating(tmp_path: Path) -> None:
    first = Store(tmp_path / "db.sqlite", window_size=5)
    first.insert_prediction("req-1", {"age": 40}, 0.5)

    second = Store(tmp_path / "db.sqlite", window_size=5)

    assert second.prediction_count() == 1
