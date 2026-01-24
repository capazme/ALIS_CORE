#!/usr/bin/env python3
"""Quick test for experts endpoint"""

import asyncio
import httpx

async def test_query():
    url = "http://localhost:8000/api/experts/query"
    payload = {
        "query": "Cos'è la legittima difesa?",
        "user_id": "test-user",
        "max_experts": 4
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"Testing: {url}")
            print(f"Payload: {payload}\n")

            response = await client.post(url, json=payload)

            print(f"Status: {response.status_code}")
            print(f"Response:\n{response.text}\n")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")
                print(f"Trace ID: {data.get('trace_id', 'N/A')}")
                print(f"Mode: {data.get('mode', 'N/A')}")
                print(f"Experts: {data.get('experts_used', [])}")
                print(f"Synthesis: {data.get('synthesis', 'N/A')[:200]}...")
            else:
                print(f"❌ FAILED: {response.status_code}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_query())
