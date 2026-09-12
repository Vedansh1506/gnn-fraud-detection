"""Create or update a dashboard user.

    uv run python -m src.db.seed_users --username analyst --role analyst
    uv run python -m src.db.seed_users --username ops --role operator

The password is read from the SEED_PASSWORD environment variable, or prompted
for - never passed as an argument, which would leave it in shell history and
process listings. Only the bcrypt hash is stored.
"""

from __future__ import annotations

import argparse
import getpass
import os

from sqlalchemy import select

from src.api.auth import hash_password
from src.db.models import ROLES, User
from src.db.session import create_tables, session_scope


def upsert_user(username: str, password: str, role: str) -> str:
    """Returns "created" or "updated" so the caller can say which happened."""
    create_tables()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None:
            session.add(
                User(username=username, password_hash=hash_password(password), role=role)
            )
            return "created"
        user.password_hash = hash_password(password)
        user.role = role
        return "updated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    args = parser.parse_args()

    password = os.environ.get("SEED_PASSWORD") or getpass.getpass("Password: ")
    if not password:
        raise SystemExit("A password is required.")

    action = upsert_user(args.username, password, args.role)
    print(f"{action} user '{args.username}' with role '{args.role}'")


if __name__ == "__main__":
    main()
