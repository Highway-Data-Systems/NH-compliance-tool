from __future__ import annotations

import socket
import subprocess
import shutil
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


APP_NAME = "NH Ride and MPD Evaluator"
HOST = "127.0.0.1"
START_PORT = 8501


def resource_path(relative_path: str) -> Path:
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / relative_path)
    candidates.append(Path(__file__).resolve().parent / relative_path)
    candidates.append(Path(sys.executable).resolve().parent / relative_path)
    candidates.append(Path.cwd() / relative_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_free_port(start_port: int = START_PORT) -> int:
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((HOST, port)) != 0:
                return port
    raise RuntimeError("Could not find a free local port for Streamlit.")


def wait_for_streamlit(url: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return 200 <= response.status < 500
        except Exception:
            time.sleep(0.5)
    return False


def run_streamlit_child(app_path: str, port: str) -> int:
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.address",
        HOST,
        "--server.port",
        port,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return streamlit_cli.main()


def streamlit_command(app_path: Path, port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        python_exe = shutil.which("python") or shutil.which("py")
        if python_exe:
            if Path(python_exe).name.lower() == "py.exe":
                return [
                    python_exe,
                    "-m",
                    "streamlit",
                    "run",
                    str(app_path),
                    "--server.address",
                    HOST,
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ]
            return [
                python_exe,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.address",
                HOST,
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
        return [sys.executable, "--run-streamlit", str(app_path), str(port)]
    return [sys.executable, str(Path(__file__).resolve()), "--run-streamlit", str(app_path), str(port)]


def main() -> int:
    app_path = resource_path("app.py")
    if not app_path.exists():
        print(f"Could not find app.py at {app_path}")
        print("Rebuild the EXE with: pyinstaller --clean app_launcher.spec")
        return 1

    port = find_free_port()
    url = f"http://{HOST}:{port}"
    command = streamlit_command(app_path, port)

    print(f"Starting {APP_NAME} at {url}")
    process = subprocess.Popen(command, cwd=str(app_path.parent))
    try:
        if wait_for_streamlit(url):
            webbrowser.open(url)
            print("The app is running. Close this window to stop it.")
        else:
            print("Streamlit did not start in time. Check the console output above.")
            return 1

        while process.poll() is None:
            time.sleep(1)
        return process.returncode or 0
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--run-streamlit":
        raise SystemExit(run_streamlit_child(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
