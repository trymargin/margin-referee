"""`--receipt` writes an unsigned receipt; `verify` checks a signed one."""
import base64
import copy
import json
import sys

import pytest

from margin_referee import cli, receipt

from test_hurdle import suite

V = {"decision": "ship", "cases": 120, "lost": 0, "won": 2, "candidates_tried": 1,
     "quality_gap": {"estimate_points": 1.7, "worst_case_points": -0.4,
                     "bar_allows_points": -2.9}}


def _statement():
    return receipt.statement(V, baseline="a", candidate="b", issuer="someone",
                             assurance_level="run-by-margin",
                             created_at="2026-11-26T00:00:00+00:00")


def _signed(tmp_path):
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization
    key = ed.Ed25519PrivateKey.generate()
    st = _statement()
    sig = key.sign(receipt.pae(receipt.PAYLOAD_TYPE, receipt.payload_bytes(st)))
    env = receipt.envelope(st, ({"keyid": "k", "sig": base64.b64encode(sig).decode()},))
    pub = base64.b64encode(key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    f = tmp_path / "receipt.json"
    f.write_text(json.dumps(env))
    return f, env, pub, st


def test_hurdle_writes_an_unsigned_self_run_receipt_and_verify_says_unsigned(tmp_path, capsys):
    root = tmp_path / ".hurdle" / "runs"
    for n, r in enumerate(suite(cost=True)):
        d = root / f"run_{n:05d}" / "revisions"
        d.mkdir(parents=True)
        (d / "000002.json").write_text(json.dumps(r))
    out = tmp_path / "receipt.json"
    assert cli.main(["hurdle", str(tmp_path / ".hurdle"), "--compare", "condition",
                     "--baseline", "baseline", "--candidate", "with-skill",
                     "--receipt", str(out)]) == 0
    printed = capsys.readouterr()
    assert "UNSIGNED" in printed.err
    verdict = json.loads(printed.out)
    env = json.loads(out.read_text())
    assert env["signatures"] == [] and env["payloadType"] == "application/vnd.in-toto+json"
    p = receipt.decode(env)["predicate"]
    assert p["assuranceLevel"] == "self-run" and p["issuer"] == "self"
    assert p["decision"] == verdict["decision"] == "ship"
    assert p["paired"]["cases"] == verdict["cases"] and p["decisionRule"]["bar"] == 0.97
    assert any("grader" in c for c in p["nonClaims"])

    assert cli.main(["verify", str(out)]) == 1, "an unsigned receipt never verifies"
    shown = json.loads(capsys.readouterr().out)
    assert shown["verified"] is False and shown["signed"] is False
    assert shown["reason"].startswith("UNSIGNED") and shown["statement"]["predicate"] == p


def test_verify_accepts_a_good_signature_and_prints_the_statement(tmp_path, capsys):
    f, _, pub, st = _signed(tmp_path)
    assert cli.main(["verify", str(f), "--key", pub]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["verified"] is True and shown["statement"] == st


def test_verify_refuses_an_edited_receipt_and_does_not_print_it(tmp_path, capsys):
    f, env, pub, st = _signed(tmp_path)
    forged = copy.deepcopy(st)
    forged["predicate"]["decision"] = "do_not_ship"
    env["payload"] = base64.b64encode(receipt.payload_bytes(forged)).decode()
    f.write_text(json.dumps(env))
    assert cli.main(["verify", str(f), "--key", pub]) == 1
    shown = json.loads(capsys.readouterr().out)
    assert shown["verified"] is False and "statement" not in shown
    assert "does not match" in shown["reason"]


def test_verify_without_a_key_or_with_a_plain_http_key_url_checks_nothing(tmp_path, capsys):
    f, _, _, _ = _signed(tmp_path)
    assert cli.main(["verify", str(f)]) == 2
    assert cli.main(["verify", str(f), "--key-url", "http://example.test/key"]) == 2
    assert "verified" not in capsys.readouterr().out


def test_verify_reads_the_key_from_an_https_url(tmp_path, capsys, monkeypatch):
    import io
    import urllib.request
    f, _, pub, _ = _signed(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(
        json.dumps({"configured": True, "public_key": pub}).encode()))
    assert cli.main(["verify", str(f), "--key-url", "https://example.test/key"]) == 0
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(
        b'{"configured": false}'))
    assert cli.main(["verify", str(f), "--key-url", "https://example.test/key"]) == 2


def test_verify_says_what_to_install_when_cryptography_is_missing(tmp_path, capsys, monkeypatch):
    f, _, pub, _ = _signed(tmp_path)
    monkeypatch.setitem(sys.modules, "cryptography.exceptions", None)
    assert cli.main(["verify", str(f), "--key", pub]) == 2
    assert "margin-referee[verify]" in capsys.readouterr().err


def test_a_file_that_is_not_a_receipt_is_refused(tmp_path, capsys):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"decision": "ship"}))
    assert cli.main(["verify", str(f), "--key", "AAAA"]) == 1
    assert "not a receipt" in json.loads(capsys.readouterr().out)["reason"]
