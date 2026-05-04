"""Static in-memory user store for the demo.

NOTE: passwords are stored in plaintext and compared with ==. This is a
local demo only — do NOT copy this pattern into any real service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class User:
    username: str
    password: str
    dealership: str
    role: str
    sp_label: str


# username -> User
_USERS: Dict[str, User] = {
    "alice": User(
        username="alice",
        password="demo123",
        dealership="North Star Motors",
        role="manager",
        sp_label="northstar",
    ),
    "bob": User(
        username="bob",
        password="demo123",
        dealership="North Star Motors",
        role="analyst",
        sp_label="northstar",
    ),
    "carol": User(
        username="carol",
        password="demo123",
        dealership="Sunrise Auto Group",
        role="manager",
        sp_label="sunrise",
    ),
    "dave": User(
        username="dave",
        password="demo123",
        dealership="Sunrise Auto Group",
        role="analyst",
        sp_label="sunrise",
    ),
}


def get_user(username: str) -> Optional[User]:
    if not username:
        return None
    return _USERS.get(username.lower())


def authenticate(username: str, password: str) -> Optional[User]:
    """Return the User on credential match, else None. Plaintext compare (demo)."""
    user = get_user(username)
    if user is None:
        return None
    # Plaintext compare — demo only.
    if user.password != password:
        return None
    return user
