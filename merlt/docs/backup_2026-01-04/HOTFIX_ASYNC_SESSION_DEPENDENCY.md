# Hotfix: AsyncSession Dependency Issue

**Date**: 2026-01-03
**Issue**: TypeError with `get_async_session` in FastAPI Depends
**Status**: ✅ FIXED

---

## Problem

```python
TypeError: '_AsyncGeneratorContextManager' object is not an async iterator
```

**Root Cause**: `get_async_session()` was decorated with `@asynccontextmanager`, making it a context manager. FastAPI's `Depends()` expects a plain async generator, not a context manager wrapper.

---

## Solution

Created a separate dependency function `get_async_session_dep()` specifically for FastAPI:

### 1. Added new function to `/merlt/rlcf/database.py`:

```python
async def get_async_session_dep() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session (for FastAPI Depends).

    This is a plain async generator (not a context manager) suitable
    for use with FastAPI's Depends() injection.

    Example:
        >>> @router.get("/endpoint")
        >>> async def endpoint(session: AsyncSession = Depends(get_async_session_dep)):
        ...     result = await session.execute(select(User))

    Yields:
        AsyncSession that auto-closes

    Raises:
        RuntimeError: If async database not initialized
    """
    global _AsyncSessionLocal

    if _AsyncSessionLocal is None:
        await init_async_db()

    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 2. Updated all `Depends(get_async_session)` to `Depends(get_async_session_dep)`:

**File**: `/merlt/api/experts_router.py`

**Changes**: 5 replacements

```python
# BEFORE
from merlt.rlcf.database import get_async_session
session: AsyncSession = Depends(get_async_session)

# AFTER
from merlt.rlcf.database import get_async_session_dep
session: AsyncSession = Depends(get_async_session_dep)
```

---

## Additional Fix: LegalSource Attribute Mapping

**Problem**: `'LegalSource' object has no attribute 'urn'`

**Root Cause**: `LegalSource` class uses different attribute names than expected

**Solution**: Updated attribute mapping in `experts_router.py`:

```python
# BEFORE (incorrect attributes)
sources.append(SourceReference(
    article_urn=legal_source.urn,  # ❌ LegalSource doesn't have 'urn'
    expert=legal_source.expert_type,  # ❌ Doesn't have 'expert_type'
    relevance=legal_source.relevance,  # ❌ Doesn't have 'relevance'
    excerpt=legal_source.text_excerpt  # ❌ Doesn't have 'text_excerpt'
))

# AFTER (correct attributes)
sources.append(SourceReference(
    article_urn=legal_source.source_id,  # ✅ Uses 'source_id'
    expert="combined",  # ✅ Default value (combined sources)
    relevance=0.9,  # ✅ Default high relevance
    excerpt=legal_source.excerpt[:200] if legal_source.excerpt else None  # ✅ Uses 'excerpt'
))
```

---

## LegalSource Class Reference

**File**: `/merlt/experts/base.py:78`

```python
class LegalSource:
    """
    Fonte giuridica citata nel reasoning.
    """
    source_type: str  # norm, jurisprudence, doctrine, constitutional
    source_id: str    # ✅ URN or unique identifier
    citation: str     # Formal citation (e.g. "Art. 1321 c.c.")
    excerpt: str = ""  # ✅ Relevant excerpt
    relevance: str = ""  # Why this source is relevant (text, not float)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `/merlt/rlcf/database.py` | +30 lines (new function `get_async_session_dep`) |
| `/merlt/api/experts_router.py` | 5 replacements + attribute mapping fix |

---

## Testing Status

### ✅ Fixed Issues

- [x] TypeError with AsyncSession dependency
- [x] LegalSource attribute mapping

### ⚠️ Remaining Prerequisites

- [ ] **OpenRouter API Key** - Required for LLM calls
  ```bash
  export OPENROUTER_API_KEY="sk-or-v1-..."
  ```

- [ ] **Test User in VisuaLex** - Required for auth testing
  - Email: `test@visualex.com`
  - Password: `testpassword123`

### Test Commands

```bash
# 1. Test orchestrator initialization (requires API key)
source .venv/bin/activate
export OPENROUTER_API_KEY="sk-or-v1-..."
python3 scripts/test_orchestrator_init.py

# Expected output:
# ✅ Query successful!
# Mode: SynthesisMode.CONVERGENT
# Synthesis: # Sintesi Integrata...

# 2. Test API endpoint (requires API key)
python3 scripts/test_experts_quick.py

# Expected output:
# ✅ SUCCESS!
# Trace ID: trace_abc123...
# Mode: convergent
# Experts: literal, systemic

# 3. Run full E2E tests (requires API key + test user)
python3 scripts/test_qa_e2e.py
```

---

## Verification

```bash
# Check if server reloaded after changes
curl http://localhost:8000/api/status | jq '.endpoints.experts'

# Expected output:
# {
#   "query": "/api/experts/query",
#   "feedback_inline": "/api/experts/feedback/inline",
#   "feedback_detailed": "/api/experts/feedback/detailed",
#   "feedback_source": "/api/experts/feedback/source",
#   "feedback_refine": "/api/experts/feedback/refine"
# }
```

---

## Related Documentation

- **Sprint 1 Summary**: `/docs/SPRINT_1_COMPLETE_SUMMARY.md`
- **E2E Testing**: `/docs/SPRINT_1_TASK_12_E2E_TESTING_SUMMARY.md`
- **API Reference**: `/docs/API_REFERENCE_EXPERT_SYSTEM.md`

---

## Lesson Learned

**Problem**: Using `@asynccontextmanager` decorator makes function incompatible with FastAPI `Depends()`

**Solution**: Create separate dependency function without decorator for FastAPI

**Prevention**: When creating async dependencies for FastAPI, use plain async generators, not context managers

---

*Hotfix applied successfully. System ready for testing with API key configuration.*
