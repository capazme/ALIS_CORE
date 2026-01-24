#!/usr/bin/env python3
"""
Test Expert System API Endpoints
==================================

Test script per verificare il corretto funzionamento degli endpoint /api/experts/*.

Usage:
    # Start API server first:
    uvicorn merlt.api.visualex_bridge:app --reload --port 8000

    # Then run this test:
    python scripts/test_expert_api.py

Requirements:
    - API server running on http://localhost:8000
    - PostgreSQL database with qa_traces and qa_feedback tables
    - OPENROUTER_API_KEY environment variable set
"""

import asyncio
import httpx
import sys
from datetime import datetime


BASE_URL = "http://localhost:8000"
USER_ID = "test_user_001"


async def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✅ Health check PASSED")


async def test_expert_query():
    """Test /api/experts/query endpoint."""
    print("\n" + "="*60)
    print("TEST 2: Expert Query")
    print("="*60)

    query_data = {
        "query": "Cos'è la legittima difesa?",
        "user_id": USER_ID,
        "context": {
            "source": "test_script",
            "timestamp": datetime.utcnow().isoformat()
        },
        "max_experts": 4
    }

    print(f"\nQuery: {query_data['query']}")
    print(f"User ID: {query_data['user_id']}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/experts/query",
                json=query_data
            )
            print(f"\nStatus: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Query successful!")
                print(f"Trace ID: {result['trace_id']}")
                print(f"Mode: {result['mode']}")
                print(f"Experts used: {', '.join(result['experts_used'])}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Execution time: {result['execution_time_ms']}ms")
                print(f"\nSynthesis (first 200 chars):\n{result['synthesis'][:200]}...")
                print(f"\nSources: {len(result['sources'])} cited")

                return result['trace_id']  # Return for follow-up tests
            else:
                print(f"❌ Query failed: {response.text}")
                return None

        except httpx.ConnectError:
            print("❌ Cannot connect to API server. Make sure it's running on port 8000")
            print("   Start it with: uvicorn merlt.api.visualex_bridge:app --reload --port 8000")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


async def test_inline_feedback(trace_id: str):
    """Test /api/experts/feedback/inline endpoint."""
    print("\n" + "="*60)
    print("TEST 3: Inline Feedback (Thumbs Up)")
    print("="*60)

    if not trace_id:
        print("⏭️  Skipping (no trace_id from previous test)")
        return

    feedback_data = {
        "trace_id": trace_id,
        "user_id": USER_ID,
        "rating": 5,  # Thumbs up
        "user_authority": 0.75
    }

    print(f"Trace ID: {feedback_data['trace_id']}")
    print(f"Rating: {feedback_data['rating']} (thumbs up)")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/experts/feedback/inline",
            json=feedback_data
        )
        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Feedback submitted successfully!")
            print(f"Feedback ID: {result['feedback_id']}")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ Feedback failed: {response.text}")


async def test_detailed_feedback(trace_id: str):
    """Test /api/experts/feedback/detailed endpoint."""
    print("\n" + "="*60)
    print("TEST 4: Detailed Feedback (3 Dimensions)")
    print("="*60)

    if not trace_id:
        print("⏭️  Skipping (no trace_id from previous test)")
        return

    feedback_data = {
        "trace_id": trace_id,
        "user_id": USER_ID,
        "retrieval_score": 0.85,
        "reasoning_score": 0.90,
        "synthesis_score": 0.80,
        "comment": "Buona risposta, sintesi chiara e ben strutturata",
        "user_authority": 0.75
    }

    print(f"Trace ID: {feedback_data['trace_id']}")
    print(f"Retrieval: {feedback_data['retrieval_score']}")
    print(f"Reasoning: {feedback_data['reasoning_score']}")
    print(f"Synthesis: {feedback_data['synthesis_score']}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/experts/feedback/detailed",
            json=feedback_data
        )
        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detailed feedback submitted successfully!")
            print(f"Feedback ID: {result['feedback_id']}")
        else:
            print(f"❌ Feedback failed: {response.text}")


async def test_api_status():
    """Test /api/status endpoint to verify expert endpoints are listed."""
    print("\n" + "="*60)
    print("TEST 5: API Status (Verify Expert Endpoints)")
    print("="*60)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/status")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            status = response.json()

            if "experts" in status["endpoints"]:
                print("\n✅ Expert endpoints registered:")
                for name, path in status["endpoints"]["experts"].items():
                    print(f"  - {name}: {path}")
            else:
                print("❌ Expert endpoints not found in API status")
        else:
            print(f"❌ Status check failed: {response.text}")


async def main():
    """Run all tests in sequence."""
    print("\n" + "="*70)
    print("MERL-T Expert System API Test Suite")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    print(f"User ID: {USER_ID}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    try:
        # Test 1: Health check
        await test_health()

        # Test 2: Expert query (returns trace_id)
        trace_id = await test_expert_query()

        if trace_id:
            # Test 3: Inline feedback
            await test_inline_feedback(trace_id)

            # Test 4: Detailed feedback
            await test_detailed_feedback(trace_id)

        # Test 5: API status
        await test_api_status()

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check database: SELECT * FROM qa_traces WHERE user_id = 'test_user_001';")
        print("  2. Check feedback: SELECT * FROM qa_feedback WHERE user_id = 'test_user_001';")
        print("  3. Explore API docs: http://localhost:8000/docs")

    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
