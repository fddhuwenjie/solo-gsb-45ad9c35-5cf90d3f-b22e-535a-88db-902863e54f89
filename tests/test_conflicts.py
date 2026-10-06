"""Integration tests for GET /conflicts (same-instant multi-source conflicts)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import (
    CONFLICT_DISTANCE_M,
    KIND_POSITION,
    SnapshotStore,
    normalize_observed_at,
)
from tests.conftest import (
    AMBIG_F1A,
    AMBIG_F1B,
    AMBIG_F2,
    SINGLE_FRAGMENT,
    SINGLE_PAYLOAD,
    VDO_SENTINELS,
)

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"

MMSI = 366999999
OTHER_MMSI = 366000001

# Same MMSI as SINGLE_FRAGMENT (40.7484, -73.9835), various latitudes.
NEAR_POSITION = "!AIVDM,1,1,,A,15Mwqgh01DreE9HGDKj4gkg40000,0*49"     # ~556 m
FAR_POSITION = "!AIVDM,1,1,,A,15Mwqgh01DreE9HGE2d4gkg40000,0*3F"      # ~2402 m
THIRD_POSITION = "!AIVDM,1,1,,A,15Mwqgh01DreE9HGEmD4gkg40000,0*40"    # ~4804 m
# bracket the 1852 m threshold on either side
JUST_OVER = "!AIVDM,1,1,,A,15Mwqgh01DreE9HGDor4gkg40000,0*75"         # ~1890 m
JUST_UNDER = "!AIVDM,1,1,,A,15Mwqgh01DreE9HGDnA4gkg40000,0*47"        # ~1812 m

# A second MMSI, two sources ~3336 m apart at the same observed_at.
M2_POSITION_A = "!AIVDM,1,1,,A,15M2oP@01Dre@D0GF904gkg40000,0*27"     # -74.0, 40.80
M2_POSITION_B = "!AIVDM,1,1,,B,15M2oP@01Dre@D0GG?D4gkg40000,0*57"     # -74.0, 40.83

# expected great-circle distance between SINGLE_FRAGMENT and FAR_POSITION
SINGLE_FAR_DISTANCE_M = 2401.8104


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh snapshot database per test."""
    monkeypatch.setenv("AIS_DB_PATH", str(tmp_path / "snapshots.db"))
    monkeypatch.setattr(main, "_store", None)
    yield TestClient(main.app)
    monkeypatch.setattr(main, "_store", None)


def one(source_id: str, sentence: str) -> dict:
    return {"source_id": source_id, "sentence": sentence}


def ingest(client: TestClient, sentences: list[dict], observed_at: str = T2):
    response = client.post(
        "/ingest", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def conflicts(
    client: TestClient, source_id: str | None = None, mmsi: int | None = None
):
    params = {}
    if source_id is not None:
        params["source_id"] = source_id
    if mmsi is not None:
        params["mmsi"] = mmsi
    response = client.get("/conflicts", params=params)
    return response.status_code, response.json()


def conflict_key(row: dict) -> tuple:
    return (
        row["observed_at"],
        row["mmsi"],
        row["source_a"]["source_id"],
        row["source_b"]["source_id"],
    )


# -- basic recording -----------------------------------------------------------


def test_empty_conflict_list(client):
    status, body = conflicts(client)
    assert status == 200
    assert body == []


def test_conflict_recorded_for_two_sources_same_instant(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", FAR_POSITION)], T2)

    status, body = conflicts(client)
    assert status == 200
    assert len(body) == 1
    row = body[0]
    assert row["mmsi"] == MMSI
    assert row["observed_at"].startswith("2026-10-05T12:00:00")
    # the pair is stored canonically (source_a < source_b)
    assert row["source_a"]["source_id"] == "rx-1"
    assert row["source_b"]["source_id"] == "rx-2"
    assert row["source_a"]["longitude"] == pytest.approx(-73.9835)
    assert row["source_a"]["latitude"] == pytest.approx(40.7484)
    assert row["source_b"]["longitude"] == pytest.approx(-73.9835)
    assert row["source_b"]["latitude"] == pytest.approx(40.77)
    assert row["distance_m"] == pytest.approx(SINGLE_FAR_DISTANCE_M, abs=0.01)
    assert row["distance_m"] > CONFLICT_DISTANCE_M


def test_conflict_present_even_when_both_points_arrive_in_one_batch(client):
    status, body = ingest(
        client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T2
    )
    assert status == 200
    _, body = conflicts(client)
    assert len(body) == 1


def test_source_pair_is_canonical_regardless_of_arrival_order(client):
    # the lexicographically larger source id stores its point first
    ingest(client, [one("rx-z", FAR_POSITION)], T2)
    ingest(client, [one("rx-a", SINGLE_FRAGMENT)], T2)

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_a"]["source_id"] == "rx-a"
    assert body[0]["source_b"]["source_id"] == "rx-z"


def test_one_conflict_row_per_source_pair_with_three_sources(client):
    ingest(
        client,
        [
            one("rx-1", SINGLE_FRAGMENT),
            one("rx-2", FAR_POSITION),
            one("rx-3", THIRD_POSITION),
        ],
        T2,
    )
    _, body = conflicts(client)
    assert {
        (r["source_a"]["source_id"], r["source_b"]["source_id"]) for r in body
    } == {("rx-1", "rx-2"), ("rx-1", "rx-3"), ("rx-2", "rx-3")}
    assert all(r["distance_m"] > CONFLICT_DISTANCE_M for r in body)


# -- threshold and exclusion rules ---------------------------------------------


def test_points_within_one_nautical_mile_are_not_conflicts(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", NEAR_POSITION)], T2)
    _, body = conflicts(client)
    assert body == []


@pytest.mark.parametrize(
    "sentence, expected",
    [(JUST_UNDER, 0), (JUST_OVER, 1)],
)
def test_threshold_is_strictly_greater_than_1852_m(client, sentence, expected):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", sentence)], T2)
    _, body = conflicts(client)
    assert len(body) == expected
    if expected:
        assert body[0]["distance_m"] > 1852.0


def test_same_source_never_conflicts_with_itself(client):
    # two batches at the same time: the second stores nothing new
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-1", FAR_POSITION)], T2)
    _, body = conflicts(client)
    assert body == []


def test_different_observed_at_never_conflicts(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_POSITION)], T2)
    _, body = conflicts(client)
    assert body == []


def test_different_mmsi_never_conflicts(client):
    ingest(client, [one("rx-1", M2_POSITION_A)], T2)
    ingest(client, [one("rx-2", FAR_POSITION)], T2)
    _, body = conflicts(client)
    assert body == []


def test_null_coordinates_never_participate(client):
    # VDO_SENTINELS carries longitude=null for the same MMSI
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", VDO_SENTINELS)], T2)
    _, body = conflicts(client)
    assert body == []


def test_same_instant_resend_adds_no_conflict_record(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", FAR_POSITION)], T2)
    # identical retransmission, and a different losing point, same time
    ingest(client, [one("rx-2", FAR_POSITION)], T2)
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T2)

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_b"]["latitude"] == pytest.approx(40.77)


# -- reviews -------------------------------------------------------------------


def test_pending_review_creates_no_conflict(client):
    status, _ = client_post_review(
        client,
        T1,
        [
            one("rx-2", SINGLE_FRAGMENT),  # unambiguous, but not applied yet
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
    )
    assert status == 201
    _, body = conflicts(client)
    assert body == []


def client_post_review(client: TestClient, observed_at: str, sentences: list[dict]):
    response = client.post(
        "/reviews", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def test_ambiguous_ingest_records_no_candidate_conflicts(client):
    # another source already has a T2 point for the same MMSI
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    # the ambiguous candidates (different MMSIs) and the unambiguous rx-2
    # position (same spot as rx-1) must not create any conflict
    status, _ = ingest(
        client,
        [
            one("rx-2", SINGLE_FRAGMENT),
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
        T2,
    )
    assert status == 409
    _, body = conflicts(client)
    assert body == []


def test_late_resolution_backfills_historical_conflict_without_snapshot_change(client):
    # ambiguous batch at the older T1 with an unambiguous far position
    batch_id = client_post_review(
        client,
        T1,
        [
            one("rx-2", FAR_POSITION),
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
    )[1]["batch_id"]
    # a different source already holds the same-instant point
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    # and rx-2 has a newer T2 snapshot that the late resolution must not roll back
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T2)
    _, body = conflicts(client)
    assert body == []  # nothing stored for rx-2 at T1 yet

    response = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 3]]}
    )
    assert response.status_code == 200

    # the historical T1 conflict is now present even though the T1 position
    # did not change rx-2's /vessels snapshot (still T2)
    _, body = conflicts(client)
    assert len(body) == 1
    row = body[0]
    assert row["observed_at"].startswith("2026-10-05T11:00:00")
    assert row["source_a"]["source_id"] == "rx-1"
    assert row["source_b"]["source_id"] == "rx-2"
    assert row["distance_m"] == pytest.approx(SINGLE_FAR_DISTANCE_M, abs=0.01)
    vessels = client.get(
        "/vessels", params={"source_id": "rx-2", "mmsi": MMSI}
    ).json()
    assert vessels[0]["position"]["observed_at"].startswith("2026-10-05T12:00:00")


def test_rejected_resolution_adds_no_conflict(client):
    batch_id = client_post_review(
        client,
        T1,
        [
            one("rx-2", FAR_POSITION),
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
    )[1]["batch_id"]
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    response = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 2]]}  # forged
    )
    assert response.status_code == 422
    _, body = conflicts(client)
    assert body == []


# -- invalidation ---------------------------------------------------------------


def test_invalidation_removes_only_pairs_involving_the_point(client):
    ingest(
        client,
        [
            one("rx-1", SINGLE_FRAGMENT),
            one("rx-2", FAR_POSITION),
            one("rx-3", THIRD_POSITION),
        ],
        T2,
    )
    assert len(conflicts(client)[1]) == 3

    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T2,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_a"]["source_id"] == "rx-2"
    assert body[0]["source_b"]["source_id"] == "rx-3"
    # filtering by the invalidated source yields nothing
    assert conflicts(client, source_id="rx-1")[1] == []
    assert len(conflicts(client, source_id="rx-2")[1]) == 1


def test_invalidating_one_side_keeps_the_other_pair(client):
    # A-B conflict at T1, A-C and B-C at T2; dropping A at T2 leaves T1
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T1)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-3", THIRD_POSITION)], T2)

    response = client.post(
        "/tracks/invalidate",
        json={"source_id": "rx-1", "mmsi": MMSI, "observed_at": T2, "reason": "r"},
    )
    assert response.status_code == 200

    _, body = conflicts(client)
    assert len(body) == 1
    assert conflict_key(body[0])[0].startswith("2026-10-05T11:00:00")
    assert body[0]["source_a"]["source_id"] == "rx-1"
    assert body[0]["source_b"]["source_id"] == "rx-2"


def test_resubmitting_at_invalidated_time_does_not_restore_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", FAR_POSITION)], T2)
    assert len(conflicts(client)[1]) == 1

    assert (
        client.post(
            "/tracks/invalidate",
            json={
                "source_id": "rx-1",
                "mmsi": MMSI,
                "observed_at": T2,
                "reason": "misreport",
            },
        ).status_code
        == 200
    )
    assert conflicts(client)[1] == []

    # the same point arrives again for the invalidated time: skipped, and
    # the removed source pair must not come back
    status, body = ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert body["summary"]["decoded"] == 1
    assert conflicts(client)[1] == []


# -- filtering and ordering ------------------------------------------------------


def test_conflicts_can_be_filtered_by_mmsi_and_source(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T1)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T2)
    ingest(
        client,
        [one("rx-1", M2_POSITION_A), one("rx-9", M2_POSITION_B)],
        T2,
    )

    _, body = conflicts(client, mmsi=MMSI)
    assert {r["mmsi"] for r in body} == {MMSI}
    assert len(body) == 2

    _, body = conflicts(client, mmsi=OTHER_MMSI)
    assert len(body) == 1
    assert body[0]["source_b"]["source_id"] == "rx-9"
    assert body[0]["distance_m"] == pytest.approx(3335.85, abs=1.0)

    assert len(conflicts(client, source_id="rx-2")[1]) == 2
    assert len(conflicts(client, source_id="rx-9")[1]) == 1
    assert conflicts(client, source_id="rx-nobody")[1] == []
    assert len(conflicts(client, source_id="rx-1", mmsi=OTHER_MMSI)[1]) == 1
    assert conflicts(client, source_id="rx-2", mmsi=OTHER_MMSI)[1] == []


def test_conflicts_are_ordered_by_time_mmsi_and_source_ids(client):
    # ingest deliberately out of the expected output order
    ingest(
        client,
        [one("rx-1", M2_POSITION_A), one("rx-9", M2_POSITION_B)],
        T2,
    )
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T2)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT), one("rx-2", FAR_POSITION)], T1)

    _, body = conflicts(client)
    keys = [
        (
            row["observed_at"][:19],
            row["mmsi"],
            row["source_a"]["source_id"],
            row["source_b"]["source_id"],
        )
        for row in body
    ]
    assert keys == sorted(keys)
    assert keys == [
        ("2026-10-05T11:00:00", MMSI, "rx-1", "rx-2"),
        ("2026-10-05T12:00:00", OTHER_MMSI, "rx-1", "rx-9"),
        ("2026-10-05T12:00:00", MMSI, "rx-1", "rx-2"),
    ]


# -- persistence and migration ----------------------------------------------------


def test_conflicts_survive_restart(client, monkeypatch):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_POSITION)], T1)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", NEAR_POSITION)], T2)

    monkeypatch.setattr(main, "_store", None)  # container restart

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["observed_at"].startswith("2026-10-05T11:00:00")
    assert body[0]["distance_m"] == pytest.approx(SINGLE_FAR_DISTANCE_M, abs=0.01)


def _position_payload(payload: str = SINGLE_PAYLOAD) -> dict:
    return decode_payload(payload, 0).model_dump()


def test_legacy_snapshot_only_database_derives_conflicts_on_upgrade(tmp_path):
    """Oldest schema (vessel_state only) yields conflicts after migration."""
    path = str(tmp_path / "legacy-state.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vessel_state (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL, kind TEXT NOT NULL,
            observed_at TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        )
        """
    )
    observed = "2026-10-05T12:00:00.000000Z"
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, KIND_POSITION, observed, json.dumps(_position_payload())),
    )
    far = decode_payload(FAR_POSITION.split(",")[5], 0).model_dump()
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-2", MMSI, KIND_POSITION, observed, json.dumps(far)),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    found = store.list_conflicts()
    assert len(found) == 1
    assert found[0].source_a.source_id == "rx-1"
    assert found[0].source_b.source_id == "rx-2"
    assert found[0].distance_m == pytest.approx(SINGLE_FAR_DISTANCE_M, abs=0.01)
    # within-limit legacy points derive no conflict
    assert all(c.distance_m > 1852.0 for c in found)
    store.close()

    reopened = SnapshotStore(path)  # idempotent: still one row after restart
    assert len(reopened.list_conflicts()) == 1
    reopened.close()


def test_pre_conflict_track_database_derives_conflicts_on_upgrade(tmp_path):
    """Database with vessel_track but no conflict table is upgraded in place."""
    path = str(tmp_path / "legacy-track.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vessel_state (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL, kind TEXT NOT NULL,
            observed_at TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE vessel_track (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL,
            observed_at TEXT NOT NULL, sentence_index INTEGER NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, observed_at)
        ) WITHOUT ROWID
        """
    )
    observed = "2026-10-05T12:00:00.000000Z"
    base = json.dumps(_position_payload())
    far = json.dumps(decode_payload(FAR_POSITION.split(",")[5], 0).model_dump())
    near = json.dumps(decode_payload(NEAR_POSITION.split(",")[5], 0).model_dump())
    for source, payload in (("rx-1", base), ("rx-2", far), ("rx-3", near)):
        conn.execute(
            "INSERT INTO vessel_track VALUES (?, ?, ?, ?, ?)",
            (source, MMSI, observed, 0, payload),
        )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    found = store.list_conflicts()
    # rx-1/rx-2 ~2402 m conflicts; rx-1/rx-3 ~556 m and rx-2/rx-3 ~1846 m
    # stay out — the threshold is strictly greater than 1852 m.
    pairs = {(c.source_a.source_id, c.source_b.source_id) for c in found}
    assert pairs == {("rx-1", "rx-2")}
    assert all(c.distance_m > 1852.0 for c in found)
    store.close()


def test_invalidated_legacy_points_create_no_conflict_on_upgrade(tmp_path):
    """A pre-conflict DB whose point is already invalid derives no conflict."""
    path = str(tmp_path / "legacy-invalid.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vessel_state (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL, kind TEXT NOT NULL,
            observed_at TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE vessel_track (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL,
            observed_at TEXT NOT NULL, sentence_index INTEGER NOT NULL,
            data TEXT NOT NULL, valid INTEGER NOT NULL DEFAULT 1,
            invalid_reason TEXT, invalidated_at TEXT,
            PRIMARY KEY (source_id, mmsi, observed_at)
        ) WITHOUT ROWID
        """
    )
    observed = "2026-10-05T12:00:00.000000Z"
    conn.execute(
        "INSERT INTO vessel_track VALUES (?, ?, ?, ?, ?, 1, NULL, NULL)",
        ("rx-1", MMSI, observed, 0, json.dumps(_position_payload())),
    )
    conn.execute(
        "INSERT INTO vessel_track VALUES (?, ?, ?, ?, ?, 0, 'misreport', ?)",
        (
            "rx-2",
            MMSI,
            observed,
            0,
            json.dumps(decode_payload(FAR_POSITION.split(",")[5], 0).model_dump()),
            "2026-10-06T00:00:00.000000Z",
        ),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    assert store.list_conflicts() == []
    store.close()


# -- atomicity --------------------------------------------------------------------


def test_conflict_and_track_commit_or_roll_back_together(tmp_path, monkeypatch):
    store = SnapshotStore(str(tmp_path / "atomic.db"))
    observed = normalize_observed_at(
        datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    )
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed, _position_payload(), 0)])

    def boom(mmsi, observed_at):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(store, "_rebuild_conflict_group", boom)
    far = decode_payload(FAR_POSITION.split(",")[5], 0).model_dump()
    with pytest.raises(RuntimeError):
        store.apply_ingest([("rx-2", MMSI, KIND_POSITION, observed, far, 0)])

    # the track insert and the conflict update rolled back together
    assert store.get_track("rx-2", MMSI) == []
    assert store.list_conflicts() == []
    store.close()


def test_failed_invalidation_keeps_point_and_conflict(tmp_path, monkeypatch):
    store = SnapshotStore(str(tmp_path / "atomic.db"))
    observed = "2026-10-05T12:00:00.000000Z"
    store.apply_ingest(
        [
            ("rx-1", MMSI, KIND_POSITION, observed, _position_payload(), 0),
            (
                "rx-2",
                MMSI,
                KIND_POSITION,
                observed,
                decode_payload(FAR_POSITION.split(",")[5], 0).model_dump(),
                1,
            ),
        ]
    )
    assert len(store.list_conflicts()) == 1

    def boom(mmsi, observed_at):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(store, "_rebuild_conflict_group", boom)
    with pytest.raises(RuntimeError):
        store.invalidate_point(
            "rx-1", MMSI, observed, "r", "2026-10-06T00:00:00.000000Z"
        )

    # the marking and the conflict removal rolled back together
    assert len(store.get_track("rx-1", MMSI)) == 1
    assert len(store.list_conflicts()) == 1
    store.close()
