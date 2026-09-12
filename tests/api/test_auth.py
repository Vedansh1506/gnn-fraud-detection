"""Auth tests: hashing, token handling, and the role gate.

No database needed - authenticate_user is covered by the endpoint tests, which
have one.
"""

import datetime as dt

import jwt
import pytest
from fastapi import HTTPException

from src.api.auth import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    hash_password,
    require_api_key,
    require_operator,
    require_user,
    verify_password,
)
from src.common.config import get_settings


def test_password_round_trips():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_same_password_hashes_differently_each_time():
    """bcrypt salts per hash - identical passwords must not produce identical
    hashes, or the table leaks which users share a password."""
    assert hash_password("same") != hash_password("same")


def test_overlong_password_is_rejected_not_truncated():
    """bcrypt silently ignores bytes past 72; truncating instead of rejecting
    would let two different long passwords authenticate each other."""
    with pytest.raises(ValueError):
        hash_password("x" * (MAX_PASSWORD_BYTES + 1))


def test_token_carries_subject_and_role():
    token = create_access_token("analyst-1", "analyst")
    settings = get_settings()
    claims = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

    assert claims["sub"] == "analyst-1"
    assert claims["role"] == "analyst"
    assert "exp" in claims


def test_require_user_accepts_a_valid_bearer_token():
    claims = require_user(f"Bearer {create_access_token('analyst-1', 'analyst')}")
    assert claims["sub"] == "analyst-1"


@pytest.mark.parametrize(
    "header", [None, "", "token-without-bearer-prefix", "Bearer not-a-real-token"]
)
def test_require_user_rejects_bad_authorization_headers(header):
    with pytest.raises(HTTPException) as exc:
        require_user(header)
    assert exc.value.status_code == 401


def test_require_user_rejects_an_expired_token():
    settings = get_settings()
    expired = jwt.encode(
        {
            "sub": "analyst-1",
            "role": "analyst",
            "exp": dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(HTTPException) as exc:
        require_user(f"Bearer {expired}")
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()


def test_require_user_rejects_a_token_signed_with_another_key():
    forged = jwt.encode({"sub": "attacker", "role": "operator"}, "not-our-key", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        require_user(f"Bearer {forged}")
    assert exc.value.status_code == 401


def test_operator_gate_allows_operators_and_forbids_analysts():
    assert require_operator({"sub": "ops", "role": "operator"})["role"] == "operator"

    with pytest.raises(HTTPException) as exc:
        require_operator({"sub": "analyst-1", "role": "analyst"})
    # 403 not 401 - authenticated, just not permitted.
    assert exc.value.status_code == 403


def test_api_key_guard_accepts_the_configured_key_and_rejects_others():
    expected = get_settings().service_api_key
    require_api_key(expected)  # does not raise

    for bad in (None, "", "wrong-key"):
        with pytest.raises(HTTPException) as exc:
            require_api_key(bad)
        assert exc.value.status_code == 401
