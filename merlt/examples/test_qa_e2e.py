#!/usr/bin/env python3
"""
End-to-End Test Script for Q&A Expert System
==============================================

Tests the complete flow from frontend → VisuaLex Backend → MERL-T API → Database

Usage:
    python scripts/test_qa_e2e.py

Requirements:
    - All services running (databases, MERL-T API, VisuaLex backend)
    - Test user credentials in VisuaLex database
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

import httpx
import asyncpg
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

MERLT_API_URL = "http://localhost:8000"
VISUALEX_API_URL = "http://localhost:3001"
FRONTEND_URL = "http://localhost:5173"

# Database configuration (MERL-T PostgreSQL)
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rlcf_dev",
    "user": "postgres",
    "password": "postgres",
}

# Test credentials (must exist in VisuaLex database)
TEST_USER = {
    "email": "test@visualex.com",
    "password": "testpassword123",
}

# Test queries
TEST_QUERIES = [
    "Cos'è la legittima difesa secondo il codice penale?",
    "Quali sono i requisiti del contratto secondo il codice civile?",
    "Cosa si intende per inadempimento grave?",
]

# =============================================================================
# TEST RESULTS TRACKING
# =============================================================================

class TestResults:
    """Track test results"""
    def __init__(self):
        self.tests: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def add(self, name: str, passed: bool, duration_ms: float, details: str = ""):
        """Add test result"""
        self.tests.append({
            "name": name,
            "passed": passed,
            "duration_ms": duration_ms,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t["passed"])
        failed = total - passed
        total_duration = sum(t["duration_ms"] for t in self.tests)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration_ms": total_duration,
            "execution_time_s": (datetime.now() - self.start_time).total_seconds(),
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.summary()

        # Summary table
        table = Table(title="🧪 E2E Test Results Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Tests", str(summary["total"]))
        table.add_row("✅ Passed", f"{summary['passed']} ({summary['pass_rate']:.1f}%)")
        table.add_row("❌ Failed", str(summary["failed"]))
        table.add_row("Total Duration", f"{summary['total_duration_ms']:.0f}ms")
        table.add_row("Execution Time", f"{summary['execution_time_s']:.2f}s")

        console.print(table)

        # Detailed results
        if self.tests:
            console.print("\n📋 Detailed Results:\n")
            for i, test in enumerate(self.tests, 1):
                status = "✅ PASS" if test["passed"] else "❌ FAIL"
                console.print(f"{i}. {status} - {test['name']} ({test['duration_ms']:.0f}ms)")
                if test["details"]:
                    console.print(f"   {test['details']}")

results = TestResults()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def timer(func, *args, **kwargs):
    """Measure function execution time"""
    start = datetime.now()
    result = await func(*args, **kwargs)
    duration_ms = (datetime.now() - start).total_seconds() * 1000
    return result, duration_ms

async def verify_database_record(conn: asyncpg.Connection, trace_id: str) -> Dict[str, Any]:
    """Verify record exists in qa_traces table"""
    row = await conn.fetchrow(
        "SELECT * FROM qa_traces WHERE trace_id = $1",
        trace_id
    )
    return dict(row) if row else None

async def count_feedback_records(conn: asyncpg.Connection, trace_id: str) -> int:
    """Count feedback records for trace_id"""
    count = await conn.fetchval(
        "SELECT COUNT(*) FROM qa_feedback WHERE trace_id = $1",
        trace_id
    )
    return count

# =============================================================================
# SERVICE HEALTH CHECKS
# =============================================================================

async def test_service_health():
    """Test 1: Verify all services are running"""
    console.print("\n[bold cyan]Test 1: Service Health Checks[/bold cyan]")

    services = {
        "MERL-T API": f"{MERLT_API_URL}/api/status",
        "VisuaLex Backend": f"{VISUALEX_API_URL}/api/health",
        "Frontend": FRONTEND_URL,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in services.items():
            try:
                response, duration_ms = await timer(client.get, url)
                passed = response.status_code == 200
                results.add(
                    f"Health: {service_name}",
                    passed,
                    duration_ms,
                    f"Status: {response.status_code}"
                )

                if passed:
                    console.print(f"  ✅ {service_name}: OK ({duration_ms:.0f}ms)")
                else:
                    console.print(f"  ❌ {service_name}: FAILED (status {response.status_code})")
            except Exception as e:
                results.add(f"Health: {service_name}", False, 0, str(e))
                console.print(f"  ❌ {service_name}: UNREACHABLE - {e}")

# =============================================================================
# DIRECT MERL-T API TESTS
# =============================================================================

async def test_direct_merlt_query():
    """Test 2: Direct query to MERL-T API (bypass VisuaLex)"""
    console.print("\n[bold cyan]Test 2: Direct MERL-T Query[/bold cyan]")

    query = TEST_QUERIES[0]
    payload = {
        "query": query,
        "user_id": "test-e2e-user",
        "max_experts": 4,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response, duration_ms = await timer(
                client.post,
                f"{MERLT_API_URL}/api/experts/query",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                passed = all([
                    "trace_id" in data,
                    "synthesis" in data,
                    "mode" in data,
                    "experts_used" in data,
                    len(data["experts_used"]) > 0,
                ])

                results.add(
                    "Direct MERL-T Query",
                    passed,
                    duration_ms,
                    f"trace_id: {data.get('trace_id', 'N/A')[:8]}... | mode: {data.get('mode')} | experts: {len(data.get('experts_used', []))}"
                )

                if passed:
                    console.print(f"  ✅ Query successful ({duration_ms:.0f}ms)")
                    console.print(f"     Trace ID: {data['trace_id'][:16]}...")
                    console.print(f"     Mode: {data['mode']}")
                    console.print(f"     Experts: {', '.join(data['experts_used'])}")
                    console.print(f"     Synthesis: {data['synthesis'][:100]}...")
                    return data["trace_id"]
                else:
                    console.print(f"  ❌ Invalid response structure")
                    return None
            else:
                results.add("Direct MERL-T Query", False, duration_ms, f"Status: {response.status_code}")
                console.print(f"  ❌ Request failed: {response.status_code}")
                console.print(f"     Error: {response.text[:200]}")
                return None

        except Exception as e:
            results.add("Direct MERL-T Query", False, 0, str(e))
            console.print(f"  ❌ Exception: {e}")
            return None

# =============================================================================
# VISUALEX PROXY TESTS
# =============================================================================

async def test_visualex_login():
    """Test 3: Login to VisuaLex to get auth token"""
    console.print("\n[bold cyan]Test 3: VisuaLex Authentication[/bold cyan]")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response, duration_ms = await timer(
                client.post,
                f"{VISUALEX_API_URL}/api/auth/login",
                json=TEST_USER
            )

            if response.status_code == 200:
                data = response.json()
                token = data.get("token")
                passed = token is not None

                results.add(
                    "VisuaLex Login",
                    passed,
                    duration_ms,
                    f"User: {TEST_USER['email']}"
                )

                if passed:
                    console.print(f"  ✅ Login successful ({duration_ms:.0f}ms)")
                    console.print(f"     Token: {token[:20]}...")
                    return token
                else:
                    console.print(f"  ❌ No token in response")
                    return None
            else:
                results.add("VisuaLex Login", False, duration_ms, f"Status: {response.status_code}")
                console.print(f"  ❌ Login failed: {response.status_code}")
                console.print(f"     Note: Ensure test user exists in VisuaLex database")
                console.print(f"     Expected: {TEST_USER['email']}")
                return None

        except Exception as e:
            results.add("VisuaLex Login", False, 0, str(e))
            console.print(f"  ❌ Exception: {e}")
            return None

async def test_visualex_proxy_query(token: str):
    """Test 4: Query through VisuaLex proxy"""
    console.print("\n[bold cyan]Test 4: VisuaLex Proxy Query[/bold cyan]")

    query = TEST_QUERIES[1]
    payload = {
        "query": query,
        "max_experts": 4,
    }

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response, duration_ms = await timer(
                client.post,
                f"{VISUALEX_API_URL}/api/merlt/experts/query",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                passed = all([
                    "trace_id" in data,
                    "synthesis" in data,
                    "mode" in data,
                ])

                results.add(
                    "VisuaLex Proxy Query",
                    passed,
                    duration_ms,
                    f"trace_id: {data.get('trace_id', 'N/A')[:8]}..."
                )

                if passed:
                    console.print(f"  ✅ Proxy query successful ({duration_ms:.0f}ms)")
                    console.print(f"     Trace ID: {data['trace_id'][:16]}...")
                    console.print(f"     Mode: {data['mode']}")
                    return data["trace_id"]
                else:
                    console.print(f"  ❌ Invalid response structure")
                    return None
            else:
                results.add("VisuaLex Proxy Query", False, duration_ms, f"Status: {response.status_code}")
                console.print(f"  ❌ Request failed: {response.status_code}")
                console.print(f"     Error: {response.text[:200]}")
                return None

        except Exception as e:
            results.add("VisuaLex Proxy Query", False, 0, str(e))
            console.print(f"  ❌ Exception: {e}")
            return None

# =============================================================================
# FEEDBACK TESTS
# =============================================================================

async def test_inline_feedback(token: str, trace_id: str):
    """Test 5: Submit inline feedback (thumbs up)"""
    console.print("\n[bold cyan]Test 5: Inline Feedback Submission[/bold cyan]")

    payload = {
        "trace_id": trace_id,
        "rating": 5,  # Thumbs up
    }

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response, duration_ms = await timer(
                client.post,
                f"{VISUALEX_API_URL}/api/merlt/experts/feedback/inline",
                json=payload,
                headers=headers
            )

            passed = response.status_code == 200

            results.add(
                "Inline Feedback",
                passed,
                duration_ms,
                f"Rating: 5 (thumbs up)"
            )

            if passed:
                console.print(f"  ✅ Inline feedback submitted ({duration_ms:.0f}ms)")
                data = response.json()
                console.print(f"     Feedback ID: {data.get('feedback_id', 'N/A')}")
            else:
                console.print(f"  ❌ Request failed: {response.status_code}")
                console.print(f"     Error: {response.text[:200]}")

        except Exception as e:
            results.add("Inline Feedback", False, 0, str(e))
            console.print(f"  ❌ Exception: {e}")

async def test_detailed_feedback(token: str, trace_id: str):
    """Test 6: Submit detailed 3-dimension feedback"""
    console.print("\n[bold cyan]Test 6: Detailed Feedback Submission[/bold cyan]")

    payload = {
        "trace_id": trace_id,
        "retrieval_score": 0.8,
        "reasoning_score": 0.9,
        "synthesis_score": 0.85,
        "comment": "E2E test - Excellent reasoning and synthesis quality",
    }

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response, duration_ms = await timer(
                client.post,
                f"{VISUALEX_API_URL}/api/merlt/experts/feedback/detailed",
                json=payload,
                headers=headers
            )

            passed = response.status_code == 200

            results.add(
                "Detailed Feedback",
                passed,
                duration_ms,
                f"Scores: {payload['retrieval_score']}/{payload['reasoning_score']}/{payload['synthesis_score']}"
            )

            if passed:
                console.print(f"  ✅ Detailed feedback submitted ({duration_ms:.0f}ms)")
                data = response.json()
                console.print(f"     Feedback ID: {data.get('feedback_id', 'N/A')}")
            else:
                console.print(f"  ❌ Request failed: {response.status_code}")
                console.print(f"     Error: {response.text[:200]}")

        except Exception as e:
            results.add("Detailed Feedback", False, 0, str(e))
            console.print(f"  ❌ Exception: {e}")

# =============================================================================
# DATABASE VERIFICATION
# =============================================================================

async def test_database_verification(trace_id: str):
    """Test 7: Verify records exist in PostgreSQL"""
    console.print("\n[bold cyan]Test 7: Database Verification[/bold cyan]")

    try:
        conn = await asyncpg.connect(**DB_CONFIG)

        # Verify qa_traces record
        trace_record, duration_ms_1 = await timer(verify_database_record, conn, trace_id)

        if trace_record:
            console.print(f"  ✅ qa_traces record found ({duration_ms_1:.0f}ms)")
            console.print(f"     User ID: {trace_record.get('user_id', 'N/A')}")
            console.print(f"     Mode: {trace_record.get('synthesis_mode', 'N/A')}")
            console.print(f"     Experts: {trace_record.get('selected_experts', [])}")

            results.add(
                "DB: qa_traces record",
                True,
                duration_ms_1,
                f"Mode: {trace_record.get('synthesis_mode')}"
            )
        else:
            console.print(f"  ❌ qa_traces record NOT FOUND")
            results.add("DB: qa_traces record", False, duration_ms_1, "Record not found")

        # Verify qa_feedback records
        feedback_count, duration_ms_2 = await timer(count_feedback_records, conn, trace_id)

        if feedback_count > 0:
            console.print(f"  ✅ qa_feedback records found: {feedback_count} ({duration_ms_2:.0f}ms)")
            results.add(
                "DB: qa_feedback records",
                True,
                duration_ms_2,
                f"Count: {feedback_count}"
            )
        else:
            console.print(f"  ⚠️  No qa_feedback records found")
            results.add("DB: qa_feedback records", False, duration_ms_2, "No records")

        await conn.close()

    except Exception as e:
        console.print(f"  ❌ Database connection failed: {e}")
        results.add("DB: Connection", False, 0, str(e))

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

async def test_performance_metrics(token: str):
    """Test 8: Performance metrics (response time, throughput)"""
    console.print("\n[bold cyan]Test 8: Performance Metrics[/bold cyan]")

    query = TEST_QUERIES[2]
    payload = {"query": query, "max_experts": 4}
    headers = {"Authorization": f"Bearer {token}"}

    # Test response time
    console.print("  📊 Testing response time...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response, duration_ms = await timer(
            client.post,
            f"{VISUALEX_API_URL}/api/merlt/experts/query",
            json=payload,
            headers=headers
        )

        # Target: < 5000ms for 95th percentile
        target_ms = 5000
        passed = duration_ms < target_ms

        results.add(
            "Performance: Response Time",
            passed,
            duration_ms,
            f"Target: <{target_ms}ms"
        )

        if passed:
            console.print(f"  ✅ Response time: {duration_ms:.0f}ms (target: <{target_ms}ms)")
        else:
            console.print(f"  ⚠️  Response time: {duration_ms:.0f}ms (exceeds target {target_ms}ms)")

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

async def test_error_handling(token: str):
    """Test 9: Error handling (invalid inputs, network errors)"""
    console.print("\n[bold cyan]Test 9: Error Handling[/bold cyan]")

    headers = {"Authorization": f"Bearer {token}"}

    # Test 9a: Invalid query (too short)
    console.print("  🧪 Testing invalid query (too short)...")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response, duration_ms = await timer(
            client.post,
            f"{VISUALEX_API_URL}/api/merlt/experts/query",
            json={"query": "Hi", "max_experts": 4},  # Only 2 chars (min is 5)
            headers=headers
        )

        # Should return 400 Bad Request
        passed = response.status_code == 400

        results.add(
            "Error: Invalid input",
            passed,
            duration_ms,
            f"Expected 400, got {response.status_code}"
        )

        if passed:
            console.print(f"  ✅ Correctly rejected invalid input ({duration_ms:.0f}ms)")
        else:
            console.print(f"  ❌ Did not reject invalid input (status: {response.status_code})")

    # Test 9b: Invalid trace_id for feedback
    console.print("  🧪 Testing invalid trace_id for feedback...")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response, duration_ms = await timer(
            client.post,
            f"{VISUALEX_API_URL}/api/merlt/experts/feedback/inline",
            json={"trace_id": "invalid-trace-id-12345", "rating": 5},
            headers=headers
        )

        # Should return 404 or 400
        passed = response.status_code in [400, 404]

        results.add(
            "Error: Invalid trace_id",
            passed,
            duration_ms,
            f"Expected 400/404, got {response.status_code}"
        )

        if passed:
            console.print(f"  ✅ Correctly handled invalid trace_id ({duration_ms:.0f}ms)")
        else:
            console.print(f"  ❌ Did not handle invalid trace_id (status: {response.status_code})")

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all E2E tests in sequence"""
    console.print(Panel.fit(
        "[bold green]🧪 Q&A Expert System - End-to-End Test Suite[/bold green]\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="green"
    ))

    # Test 1: Service health
    await test_service_health()

    # Test 2: Direct MERL-T query
    trace_id_direct = await test_direct_merlt_query()

    # Test 3: VisuaLex login
    token = await test_visualex_login()

    if not token:
        console.print("\n[bold red]❌ Cannot proceed without auth token[/bold red]")
        console.print("[yellow]Note: Create test user with:[/yellow]")
        console.print(f"  Email: {TEST_USER['email']}")
        console.print(f"  Password: {TEST_USER['password']}")
        results.print_summary()
        return

    # Test 4: VisuaLex proxy query
    trace_id_proxy = await test_visualex_proxy_query(token)

    if trace_id_proxy:
        # Test 5: Inline feedback
        await test_inline_feedback(token, trace_id_proxy)

        # Test 6: Detailed feedback
        await test_detailed_feedback(token, trace_id_proxy)

        # Test 7: Database verification
        await test_database_verification(trace_id_proxy)

    # Test 8: Performance metrics
    await test_performance_metrics(token)

    # Test 9: Error handling
    await test_error_handling(token)

    # Print summary
    console.print("\n" + "="*80 + "\n")
    results.print_summary()

    # Final verdict
    summary = results.summary()
    if summary["failed"] == 0:
        console.print("\n[bold green]🎉 ALL TESTS PASSED![/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]⚠️  {summary['failed']} TESTS FAILED[/bold red]")
        return 1

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Test suite interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
