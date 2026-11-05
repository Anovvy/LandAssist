import asyncio
import json
import subprocess
import tempfile
import time
import httpx
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://127.0.0.1:8000/editor/#map=14.50/-6.2200/106.5000&background=Bing"

async def main():
    temp_dir = tempfile.mkdtemp()
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--user-data-dir={temp_dir}",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        URL
    ])
    try:
        ws_url = None
        for _ in range(30):
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.get("http://127.0.0.1:9222/json")
                    if res.status_code == 200:
                        pages = res.json()
                        for p in pages:
                            if p.get("type") == "page" and "webSocketDebuggerUrl" in p:
                                ws_url = p["webSocketDebuggerUrl"]
                                break
                        if ws_url:
                            break
            except Exception:
                pass
            time.sleep(0.3)

        if not ws_url:
            print("Failed to connect to Chrome DevTools port")
            return

        print(f"Connected to Chrome ws: {ws_url}")
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
            await ws.send(json.dumps({"id": 2, "method": "Log.enable"}))
            await ws.send(json.dumps({"id": 3, "method": "Page.enable"}))

            start = time.time()
            while time.time() - start < 10:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    method = data.get("method")
                    if method == "Runtime.exceptionThrown":
                        details = data["params"]["exceptionDetails"]
                        exp = details.get("exception", {})
                        print(f"[EXCEPTION] {details.get('text')} - {exp.get('description', exp)}")
                    elif method == "Runtime.consoleAPICalled":
                        p_type = data["params"]["type"]
                        args = [str(a.get("value", a.get("description", ""))) for a in data["params"]["args"]]
                        print(f"[CONSOLE.{p_type}] {' '.join(args)}")
                    elif method == "Log.entryAdded":
                        entry = data["params"]["entry"]
                        print(f"[LOG.{entry.get('level')}] {entry.get('text')}")
                except asyncio.TimeoutError:
                    pass
    finally:
        proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
