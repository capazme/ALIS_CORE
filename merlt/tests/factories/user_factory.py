"""Factory for generating test users with authority scores."""

import random
import uuid
from datetime import datetime, timezone, timedelta


PROFILE_TYPES = ["studente", "avvocato", "magistrato", "accademico"]

_ITALIAN_NAMES = [
    ("Marco", "Rossi"), ("Giulia", "Bianchi"), ("Alessandro", "Romano"),
    ("Francesca", "Colombo"), ("Luca", "Ricci"), ("Sara", "Marino"),
    ("Andrea", "Greco"), ("Chiara", "Bruno"), ("Matteo", "Gallo"),
    ("Elena", "Conti"), ("Giovanni", "De Luca"), ("Valentina", "Costa"),
    ("Paolo", "Giordano"), ("Simona", "Mancini"), ("Roberto", "Barbieri"),
]

_counter = 0


def _next_counter() -> int:
    global _counter
    _counter += 1
    return _counter


def create_user(**overrides) -> dict:
    """Create a single test user dict.

    Returns dict with: user_id, username, email, authority_score,
    profile_type, created_at.
    """
    n = _next_counter()
    first, last = random.choice(_ITALIAN_NAMES)
    username = f"{first.lower()}.{last.lower()}{n}"

    defaults = {
        "user_id": str(uuid.uuid4()),
        "username": username,
        "email": f"{username}@example.com",
        "authority_score": round(random.uniform(0.0, 1.0), 3),
        "profile_type": random.choice(PROFILE_TYPES),
        "created_at": (
            datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365))
        ).isoformat(),
    }
    defaults.update(overrides)
    return defaults


def create_users(count: int, **overrides) -> list[dict]:
    """Create a list of test users."""
    return [create_user(**overrides) for _ in range(count)]
