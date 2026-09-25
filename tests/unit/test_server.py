"""Tests for the REST API, against the synthetic index from conftest.py."""

import os
from dataclasses import fields
from pathlib import Path

import pytest

from mmlite.config import Settings

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from mmlite import server


@pytest.fixture(scope="module")
def client(index_dir: Path):
    settings = Settings(index_directory=index_dir, enable_postagging=False)
    server._service = server.build_service(settings)
    with TestClient(server.app) as c:  # the lifespan handler closes the service on exit
        yield c
    assert server._service is None, "lifespan shutdown should release the index"


def test_health(client, index_dir: Path):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_directory"] == str(index_dir)
    assert body["index"]["norm_version"]
    assert body["postagging"] is False and body["detect_negations"] is True
    assert body["postag_model"] is None  # tagging is off, so no model is in use
    assert body["uptime_s"] >= 0


def test_annotate(client):
    r = client.post("/annotate", json={"text": "Heart Attack and aspirin.", "docid": "d1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["docid"] == "d1"
    assert [e["text"] for e in body["entities"]] == ["Heart Attack", "aspirin"]
    first = body["entities"][0]
    assert (first["start"], first["length"]) == (0, 12)
    assert first["negated"] is False
    assert first["concepts"][0] == {
        "cui": "C0027051",
        "preferred_name": "Myocardial Infarction",
        "semantic_types": ["dsyn"],
        "sources": ["MSH"],
        "matched_string": "Heart Attack",
    }
    assert body["elapsed_ms"] >= 0


def test_annotate_restrictions_and_negation(client):
    r = client.post(
        "/annotate",
        json={"text": "Heart Attack and aspirin.", "restrict_to_sts": ["phsu"]},
    )
    assert [e["text"] for e in r.json()["entities"]] == ["aspirin"]

    r = client.post("/annotate", json={"text": "No heart attack.", "restrict_to_sources": ["MSH"]})
    assert r.json()["entities"][0]["negated"] is True

    r = client.post("/annotate", json={"text": "No heart attack.", "detect_negations": False})
    assert r.json()["entities"][0]["negated"] is False


def test_annotate_validation(client):
    assert client.post("/annotate", json={"text": "   "}).status_code == 422
    assert client.post("/annotate", json={}).status_code == 422
    big = {"text": "a" * (server.MAX_TEXT_CHARS + 1)}
    assert client.post("/annotate", json=big).status_code == 413


def test_batch(client):
    r = client.post(
        "/annotate/batch",
        json={
            "documents": [
                {"docid": "a.txt", "text": "aspirin"},
                {"docid": "b.txt", "text": "Heart Attack"},
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["docid"] for x in body["results"]] == ["a.txt", "b.txt"]
    assert body["results"][0]["entities"][0]["concepts"][0]["cui"] == "C0004057"
    assert body["elapsed_ms"] >= 0


def test_batch_validation(client):
    assert client.post("/annotate/batch", json={"documents": []}).status_code == 422
    many = {"documents": [{"text": "aspirin"}] * (server.MAX_BATCH_DOCS + 1)}
    assert client.post("/annotate/batch", json=many).status_code == 413


def test_annotate_text_formats(client):
    r = client.post("/annotate/text", json={"text": "Heart Attack.", "docid": "d.txt"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text.startswith("d.txt|MMI|") and r.text.rstrip().endswith("|")

    r = client.post("/annotate/text?format=cuilist", json={"text": "Heart Attack."})
    assert r.text == "C0027051\n"

    r = client.post("/annotate/text?format=brat", json={"text": "Heart Attack."})
    assert r.text.startswith("T1\tMMLite 0 12\tHeart Attack")

    assert client.post("/annotate/text?format=nope", json={"text": "x"}).status_code == 422


def test_lookup(client):
    r = client.get("/lookup", params={"term": "Alzheimer's disease"})
    assert r.status_code == 200, r.text
    assert [c["cui"] for c in r.json()] == ["C0002395"]
    assert client.get("/lookup", params={"term": " "}).status_code == 422
    assert client.get("/lookup", params={"term": "zzz no such term"}).json() == []


def test_every_index_touching_endpoint_holds_the_service_lock(client, monkeypatch):
    """The pipeline objects are not thread-safe, so no endpoint may reach past Service.lock."""
    held: list[bool] = []
    service = server._service
    real = service.mml.__class__

    for name in ("process_text", "process_document", "lookup_term", "format"):
        original = getattr(real, name)

        def wrapper(self, *a, _original=original, **kw):
            held.append(service.lock.locked())
            return _original(self, *a, **kw)

        monkeypatch.setattr(real, name, wrapper)

    client.post("/annotate", json={"text": "aspirin"})
    client.post("/annotate/batch", json={"documents": [{"text": "aspirin"}]})
    client.post("/annotate/text", json={"text": "aspirin"})
    client.post("/annotate/formatted", json={"text": "aspirin", "inputformat": "freetext"})
    client.get("/lookup", params={"term": "aspirin"})
    assert held and all(held), f"an endpoint reached the index unlocked: {held}"


def test_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "mmlite"
    assert set(schema["paths"]) == {
        "/health",
        "/formats",
        "/annotate",
        "/annotate/batch",
        "/annotate/text",
        "/annotate/formatted",
        "/lookup",
        "/semantic-types",
    }


def test_export_settings_round_trips_every_field(tmp_path: Path, monkeypatch):
    """`serve --reload` hands settings to the worker through the environment.

    Nothing may be lost on the way.  Every field is set to a non-default value, and the test
    checks that first: a field left at its default would round-trip whether or not it was
    exported, which is how ``negation_detector`` went missing unnoticed.
    """
    original = Settings(
        index_directory=tmp_path / "ivf",
        sourceset=["MSH", "SNOMEDCT_US"],
        semantic_types=["dsyn", "sosy"],
        excluded_terms_file=tmp_path / "special.txt",
        uda_file=tmp_path / "uda.txt",
        cui_term_list_file=tmp_path / "cuis.txt",
        segmentation_method="LINES",
        enable_postagging=False,
        postag_list=["NN", "JJ"],
        postag_model="en_core_sci_sm",
        postag_verb_rescue=0.5,
        detect_negations=False,
        negation_detector="context",
        strict_parity=False,
        precision_filter=True,
        brat_typename="Concept",
        normalized_string_cache_size=4242,
    )
    defaults = Settings()
    left_at_default = [
        f.name for f in fields(Settings) if getattr(original, f.name) == getattr(defaults, f.name)
    ]
    assert not left_at_default, f"set these to non-default values: {left_at_default}"
    for name in list(os.environ):
        if name.startswith("MMLITE_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    server._export_settings(original)
    assert Settings.load() == original


def test_service_unavailable_without_index(monkeypatch):
    monkeypatch.setattr(server, "_service", None)
    with pytest.raises(fastapi.HTTPException) as e:
        server.get_service()
    assert e.value.status_code == 503


def test_formats_lists_everything_the_server_accepts(client):
    body = client.get("/formats").json()
    assert "freetext" in body["input_formats"] and "chemdner" in body["input_formats"]
    assert "mmi" in body["output_formats"] and "bc" in body["output_formats"]
    assert set(body["negation_detectors"]) == {"negex", "context"}
    assert "SENTENCES" in body["segmentation_methods"]


def test_semantic_types_are_listed_for_building_restrictions(client):
    body = client.get("/semantic-types").json()
    assert {"tui": "T047", "abbreviation": "dsyn", "name": "Disease or Syndrome"} in body
    assert len(body) > 100


def test_annotate_formatted_parses_an_input_format_and_annotates_each_document(client):
    r = client.post(
        "/annotate/formatted",
        json={
            "text": "D1\tHeart attack study.\tThe patient has diabetes.\nD2\tAspirin.\tNo fever.\n",
            "inputformat": "chemdner",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [res["docid"] for res in body["results"]] == ["D1", "D2"]
    assert any(
        c["cui"] == "C0011849"
        for res in body["results"]
        for e in res["entities"]
        for c in e["concepts"]
    )


def test_annotate_formatted_rejects_too_many_documents_before_annotating_any(client, monkeypatch):
    """The count used to be checked on the *result*: every document was annotated under the
    service lock first, then the whole batch thrown away with a 413."""
    service = server._service
    annotated: list[str] = []
    real = service.mml.process_document

    def counting(document, **kw):
        annotated.append(document.resolved_id())
        return real(document, **kw)

    monkeypatch.setattr(service.mml, "process_document", counting)
    n = server.MAX_BATCH_DOCS + 1
    text = "".join(f"D{i}\tAspirin.\tNo fever.\n" for i in range(n))
    r = client.post("/annotate/formatted", json={"text": text, "inputformat": "chemdner"})
    assert r.status_code == 413
    assert f"input yields {n} documents" in r.json()["detail"]
    assert annotated == []

    ok = client.post(
        "/annotate/formatted",
        json={"text": text.split("\n", 1)[0] + "\n", "inputformat": "chemdner"},
    )
    assert ok.status_code == 200 and annotated == ["D0"]  # the counter itself works


def test_text_limit_is_per_document_and_the_formatted_body_has_its_own(client, monkeypatch):
    """MAX_TEXT_CHARS bounds one document (the thing whose length drives latency), so a formatted
    body of many small documents may exceed it, but no single document inside one may."""
    assert server.MAX_TEXT_CHARS == 50_000 and server.MAX_FORMATTED_CHARS == 1_000_000
    service = server._service
    annotated: list[str] = []
    real = service.mml.process_document
    monkeypatch.setattr(
        service.mml,
        "process_document",
        lambda d, **kw: annotated.append(d.resolved_id()) or real(d, **kw),
    )
    monkeypatch.setattr(server, "MAX_TEXT_CHARS", 1_000)

    small = "".join(f"D{i}\tAspirin.\t{'No fever. ' * 50}\n" for i in range(5))  # ~2.6 KB total
    assert len(small) > server.MAX_TEXT_CHARS
    ok = client.post("/annotate/formatted", json={"text": small, "inputformat": "chemdner"})
    assert ok.status_code == 200, ok.text

    annotated.clear()
    big = "D0\tAspirin.\tNo fever.\nD1\tAspirin.\t" + "No fever. " * 200 + "\n"
    r = client.post("/annotate/formatted", json={"text": big, "inputformat": "chemdner"})
    assert r.status_code == 413 and "document 'D1' exceeds 1000 characters" in r.json()["detail"]
    assert annotated == []  # rejected before any document was annotated

    monkeypatch.setattr(server, "MAX_FORMATTED_CHARS", 100)
    r = client.post("/annotate/formatted", json={"text": small, "inputformat": "chemdner"})
    assert r.status_code == 413 and "text exceeds 100 characters" in r.json()["detail"]


def test_annotate_formatted_rejects_an_unknown_input_format(client):
    r = client.post("/annotate/formatted", json={"text": "x", "inputformat": "nope"})
    assert r.status_code == 422 and "unknown input format" in r.json()["detail"]


def test_annotate_formatted_reports_malformed_xml_as_a_client_error(client):
    r = client.post(
        "/annotate/formatted",
        json={"text": "<PubmedArticleSet><PubmedArticle>", "inputformat": "pubmed"},
    )
    assert r.status_code == 422 and "not well-formed XML" in r.json()["detail"]


def test_detector_can_be_chosen_per_request_and_is_restored_afterwards(client):
    service = server._service
    before = service.mml.lookup.negex
    r = client.post("/annotate", json={"text": "Her mother had diabetes.", "detector": "context"})
    assert r.status_code == 200, r.text
    entities = r.json()["entities"]
    assert entities and all(e["temporality"] for e in entities)
    assert any(e["experiencer"] == "Other" for e in entities)
    # the server is back on its configured detector, which reports neither field
    assert service.mml.lookup.negex is before
    plain = client.post("/annotate", json={"text": "Her mother had diabetes."}).json()["entities"]
    assert all(e["temporality"] is None for e in plain)


def test_an_unknown_detector_is_a_client_error_and_leaves_the_server_alone(client):
    service = server._service
    before = service.mml.lookup.negex
    r = client.post("/annotate", json={"text": "diabetes", "detector": "nope"})
    assert r.status_code == 422 and "unknown negation detector" in r.json()["detail"]
    assert service.mml.lookup.negex is before


def test_health_reports_the_configured_detector(client):
    assert client.get("/health").json()["negation_detector"] == "negex"


def test_health_reports_whether_output_follows_java_exactly(client):
    """strict_parity=False changes ConText results and output ordering, so a caller comparing
    runs needs to know which mode produced them."""
    assert client.get("/health").json()["strict_parity"] is True


def test_strict_context_service_warns_once_at_start_up(index_dir: Path, caplog):
    """Strict ConText inverts experiencer/temporality across sentences and is quadratic, so a
    service configured with it says so; fixed mode and NegEx stay quiet."""
    for detector, strict, warns in (
        ("context", True, True),
        ("context", False, False),
        ("negex", True, False),
    ):
        caplog.clear()
        settings = Settings(
            index_directory=index_dir,
            enable_postagging=False,
            negation_detector=detector,
            strict_parity=strict,
        )
        service = server.build_service(settings)
        try:
            assert (server.STRICT_CONTEXT_WARNING in caplog.text) is warns, (detector, strict)
        finally:
            service.mml.close()


def test_strict_context_chosen_per_request_warns_once(client, caplog):
    server._service._warned_strict_context = False
    for _ in range(2):
        r = client.post(
            "/annotate", json={"text": "Her mother had diabetes.", "detector": "context"}
        )
        assert r.status_code == 200, r.text
    assert caplog.text.count(server.STRICT_CONTEXT_WARNING) == 1


def test_bearer_token_guards_every_endpoint_but_health(client, monkeypatch):
    monkeypatch.setattr(server, "API_TOKEN", "s3cret")
    guarded = [
        ("post", "/annotate", {"json": {"text": "aspirin"}}),
        ("post", "/annotate/batch", {"json": {"documents": [{"text": "aspirin"}]}}),
        ("post", "/annotate/text", {"json": {"text": "aspirin"}}),
        ("post", "/annotate/formatted", {"json": {"text": "aspirin"}}),
        ("get", "/lookup", {"params": {"term": "aspirin"}}),
        ("get", "/formats", {}),
        ("get", "/semantic-types", {}),
    ]
    for method, path, kwargs in guarded:
        call = getattr(client, method)
        r = call(path, **kwargs)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "Bearer"
        r = call(path, headers={"Authorization": "Bearer wrong"}, **kwargs)
        assert r.status_code == 401, path
        r = call(path, headers={"Authorization": "Bearer s3cret"}, **kwargs)
        assert r.status_code == 200, (path, r.text)
    health = client.get("/health")  # liveness probes carry no credentials
    assert health.status_code == 200 and health.json()["auth_required"] is True


def test_no_token_configured_leaves_the_api_open(client):
    assert server.API_TOKEN is None
    assert client.get("/formats").status_code == 200
    assert client.get("/health").json()["auth_required"] is False


def test_cors_is_off_by_default_and_configurable(monkeypatch):
    monkeypatch.delenv("MMLITE_CORS_ORIGINS", raising=False)
    assert server._cors_origins() == []
    monkeypatch.setenv("MMLITE_CORS_ORIGINS", " https://a.example , https://b.example,")
    assert server._cors_origins() == ["https://a.example", "https://b.example"]


def test_cors_origins_from_the_environment_reach_the_app():
    """The middleware is added at import, so check a fresh interpreter."""
    import subprocess
    import sys

    code = (
        "from fastapi.testclient import TestClient\n"
        "from mmlite import server\n"
        "r = TestClient(server.app).options('/formats', headers={\n"
        "    'Origin': 'https://ui.example', 'Access-Control-Request-Method': 'GET'})\n"
        "print(r.headers.get('access-control-allow-origin'))\n"
    )
    env = {**os.environ, "MMLITE_CORS_ORIGINS": "https://ui.example"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "https://ui.example"
