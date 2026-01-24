"""
Test script for /graph/search endpoint.

Tests the semantic graph search endpoint with various queries and filters.
"""

import asyncio
import httpx
import os
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


async def test_graph_search():
    """Test the graph search endpoint."""
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

    # Test cases
    test_cases = [
        {
            "name": "Basic search - responsabilità",
            "payload": {
                "query": "responsabilità del debitore",
                "limit": 5,
            },
        },
        {
            "name": "Filtered search - only principio",
            "payload": {
                "query": "buona fede contrattuale",
                "filters": {
                    "entity_types": ["principio"]
                },
                "limit": 3,
            },
        },
        {
            "name": "Filtered search - specific relations",
            "payload": {
                "query": "inadempimento contratto",
                "filters": {
                    "relation_types": ["DISCIPLINA", "ESPRIME_PRINCIPIO"]
                },
                "limit": 5,
            },
        },
        {
            "name": "Complex query",
            "payload": {
                "query": "risoluzione del contratto per inadempimento",
                "filters": {
                    "entity_types": ["principio", "concetto"]
                },
                "limit": 10,
            },
        },
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, test_case in enumerate(test_cases, 1):
            console.rule(f"[bold blue]Test {i}: {test_case['name']}")

            try:
                response = await client.post(
                    f"{base_url}/graph/search",
                    json=test_case["payload"]
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display results
                    subgraph = data.get("subgraph", {})
                    nodes = subgraph.get("nodes", [])
                    edges = subgraph.get("edges", [])
                    relevance_scores = data.get("relevance_scores", {})
                    metadata = subgraph.get("metadata", {})

                    console.print(f"[green]✓ Success")
                    console.print(f"  Query time: {data.get('query_time_ms', 0):.2f} ms")
                    console.print(f"  Nodes found: {len(nodes)}")
                    console.print(f"  Edges found: {len(edges)}")

                    if nodes:
                        # Create table with top results
                        table = Table(title="Top Results", show_lines=True)
                        table.add_column("Node ID", style="cyan", max_width=50)
                        table.add_column("Type", style="magenta")
                        table.add_column("Label", style="white", max_width=60)
                        table.add_column("Relevance", style="green", justify="right")

                        for node in nodes[:5]:  # Top 5
                            node_id = node.get("id", "")
                            relevance = relevance_scores.get(node_id, 0.0)
                            table.add_row(
                                node_id,
                                node.get("type", ""),
                                node.get("label", "")[:60],
                                f"{relevance:.3f}"
                            )

                        console.print(table)

                    if edges:
                        console.print(f"\n[yellow]Sample relations:")
                        for edge in edges[:3]:
                            console.print(
                                f"  • {edge.get('source', '')} "
                                f"[bold]-[{edge.get('type', '')}]->[/bold] "
                                f"{edge.get('target', '')}"
                            )
                else:
                    console.print(f"[red]✗ Failed: HTTP {response.status_code}")
                    console.print(f"  {response.text}")

            except Exception as e:
                console.print(f"[red]✗ Error: {e}")

            console.print()  # Blank line


async def test_empty_results():
    """Test with query that should return no results."""
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

    console.rule("[bold blue]Test: Empty Results")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{base_url}/graph/search",
                json={
                    "query": "xyzabc123notexistingatall",
                    "limit": 5,
                }
            )

            if response.status_code == 200:
                data = response.json()
                nodes = data.get("subgraph", {}).get("nodes", [])

                if len(nodes) == 0:
                    console.print("[green]✓ Correctly returned empty results")
                else:
                    console.print(f"[yellow]⚠ Expected 0 nodes, got {len(nodes)}")
            else:
                console.print(f"[red]✗ Failed: HTTP {response.status_code}")

        except Exception as e:
            console.print(f"[red]✗ Error: {e}")


async def main():
    """Run all tests."""
    console.print("[bold cyan]Graph Search Endpoint Test Suite\n")

    # Check if API is running
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url.replace('/api/v1', '')}/health")
            if response.status_code != 200:
                console.print("[red]⚠ API not running at {base_url}")
                console.print("Please start the API server first.")
                return
    except Exception:
        console.print(f"[red]⚠ Cannot connect to API at {base_url}")
        console.print("Please start the API server first.")
        return

    await test_graph_search()
    await test_empty_results()

    console.rule("[bold green]Tests completed")


if __name__ == "__main__":
    asyncio.run(main())
