"""Quick smoke test for imports."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from algochat_base import discover_algorithms, ALGORITHMS
discover_algorithms()
print(f"Algorithms discovered: {len(ALGORITHMS)}")

from tools import build_tools, execute_algorithm
tools = build_tools()
print(f"Tools built: {len(tools)}")
for t in tools:
    name = t["function"]["name"]
    desc = t["function"]["description"][:80]
    params_count = len(t["function"]["parameters"].get("properties", {}))
    print(f"  - {name}: params={params_count} | {desc}...")

from llm_client import get_llm_client
llm = get_llm_client()
print(f"LLM configured: {llm.configured}")

from agent import get_agent
agent = get_agent()
print(f"Agent can run: {agent.can_run()}")
print(f"Agent tools count: {len(agent._tools)}")

print("\nAll imports OK!")