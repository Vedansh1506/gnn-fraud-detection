"""Authentication for the two MVP personas (SAD 8).

Two mechanisms, matching TRD 5.2:
- **X-API-Key** for service-to-service calls (the stream consumer -> /score).
- **JWT** for user-facing calls (the dashboard -> everything else), carrying a
  role claim so /retrain can be operator-only.

Enterprise SSO/RBAC is explicitly out of scope, so this is one role string per
user rather than a permissions model. The documented production upgrade is
AWS Cognito.

Passwords are hashed with bcrypt directly rather than through passlib: passlib
1.7.4 (last released 2020) raises on import against bcrypt 5.x. Same algorithm,
maintained wrapper.
"""

from __future__ import annotations

import datetime as dt
import secrets
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select

from src.common.config import get_settings
from src.db.models import OPERATOR_ROLE, User
from src.db.session import session_scope

# bcrypt silently truncates beyond 72 bytes, so reject rather than let two
# different long passwords authenticate each other.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    _reject_overlong(password)
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _reject_overlong(password: str) -> None:
    if len(password.encode()) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")


def create_access_token(username: str, role: str) -> str:
    settings = get_settings()
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expires},
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def authenticate_user(username: str, password: str) -> User | None:
    """None on either unknown user or bad password - the caller must not
    distinguish the two, or it leaks which usernames exist."""
    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None:
            # Hash anyway so a missing user doesn't return measurably faster
            # than a wrong password (timing side channel).
            verify_password(password, _DUMMY_HASH)
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Service-to-service guard. secrets.compare_digest to keep the comparison
    constant-time."""
    expected = get_settings().service_api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key"
        )



def require_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1]
    settings = get_settings()
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc


def require_operator(claims: Annotated[dict, Depends(require_user)]) -> dict:
    """403 not 401: the caller is authenticated, just not allowed."""
    if claims.get("role") != OPERATOR_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Operator role required"
        )
    return claims
