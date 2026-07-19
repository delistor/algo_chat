"""Integration test for the ReAct Agent."""
import urllib.request, urllib.parse, json, ssl, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ctx = ssl.create_default_context()
BASE = "http://localhost:8000"

def test_agent(message, session_id="test_session"):
    data = urllib.parse.urlencode({
        "message": message,
        "conversation_id": session_id,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat/agent",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=180)
        body = resp.read().decode()
        print(f"Status: {resp.status}\n")

        events = []
        for line in body.strip().split("\n"):
            if line.startswith("data: "):
                evt = json.loads(line[6:])
                events.append(evt)

        # Summarize
        evt_types = {}
        for e in events:
            t = e.get("event", "content_delta")
            evt_types[t] = evt_types.get(t, 0) + 1

        print(f"Event counts: {evt_types}")

        # Show tool calls
        for e in events:
            if e.get("event") == "tool_call":
                print(f"\n>>> TOOL_CALL: {e.get('name', '?')}")
                print(f"    args: {json.dumps(e.get('arguments', {}), ensure_ascii=False)[:300]}")
            elif e.get("event") == "tool_result":
                res = e.get("result", {})
                if isinstance(res, dict):
                    print(f"\n<<< TOOL_RESULT keys: {list(res.keys())[:8]}")
                    for k, v in res.items():
                        if k != "image":
                            val_str = str(v)[:120]
                            print(f"    {k}: {val_str}")
                else:
                    print(f"\n<<< TOOL_RESULT: {str(res)[:200]}")
            elif e.get("event") == "done":
                print(f"\n=== AGENT DONE ===")
            elif e.get("event") == "error":
                print(f"\n!!! ERROR: {e.get('message', str(e))}")

        # Reconstruct full content
        content = "".join(e.get("content", "") for e in events)
        print(f"\n--- Agent Response ---\n{content}\n---")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "用数据统计工具分析一个随机生成的CSV文件，10个样本，3个特征列"
    sid = sys.argv[2] if len(sys.argv) > 2 else "test_integration"
    ok = test_agent(msg, sid)
    sys.exit(0 if ok else 1)