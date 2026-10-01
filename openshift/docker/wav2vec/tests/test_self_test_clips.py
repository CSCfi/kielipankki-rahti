"""The self test reports SKIPPED checks rather than failing when the clips
are not bundled."""

from asr import api, config


def test_has_test_clips(tmp_path):
    assert not api.has_test_clips(str(tmp_path))
    for name in api.TEST_CLIPS:
        (tmp_path / name).write_bytes(b"")
    assert api.has_test_clips(str(tmp_path))


def test_self_test_without_clips(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TEST_DATA_DIR", str(tmp_path))

    class Health:
        def json(self):
            return {"status": "UP", "checks": {"redis": "UP"}}

    monkeypatch.setattr(api.requests, "get", lambda *a, **k: Health())
    api.app.config["TESTING"] = True
    r = api.app.test_client().get("/audio/asr/fi/self_test").get_json()
    assert r["status"] == "UP" and r["test_clips"] is None
    assert r["checks"]["align"] == "SKIPPED" and r["checks"]["redis"] == "UP"
