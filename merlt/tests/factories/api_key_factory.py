"""Factory for generating API key test data."""

import hashlib
import random
import uuid
from datetime import datetime, timezone, timedelta


ROLES = ["admin", "user", "guest"]
TIERS = ["unlimited", "premium", "standard", "limited"]


def create_api_key(**overrides) -> dict:
    """Create an API key test dict.

    Returns dict with: key_id, key_hash, name, role, tier,
    is_active, created_at, expires_at, user_id, description.
    """
    raw_key = f"merlt_{uuid.uuid4().hex}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    defaults = {
        "key_id": f"key_{uuid.uuid4().hex[:12]}",
        "api_key_hash": key_hash,
        "raw_key": raw_key,  # Only available in tests, never stored in prod
        "role": random.choice(ROLES),
        "rate_limit_tier": random.choice(TIERS),
        "is_active": True,
        "user_id": str(uuid.uuid4()),
        "description": "Test API key",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(days=random.randint(30, 365))
        ).isoformat(),
    }
    defaults.update(overrides)
    return defaults
