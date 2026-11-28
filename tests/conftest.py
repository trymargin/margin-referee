"""No test here opens a socket: the key balance read fails closed to "balance not checked"."""
import pytest

from margin_referee import endpoints


@pytest.fixture(autouse=True)
def no_balance_read(monkeypatch):
    def offline(url, headers):
        raise OSError("offline in tests")
    monkeypatch.setattr(endpoints, "_get", offline)
