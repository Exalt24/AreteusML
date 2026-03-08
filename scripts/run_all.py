"""Launch all AreteusML services in a single terminal.

Usage:
    python scripts/run_all.py              # API + Streamlit + Dagster + MLflow
    python scripts/run_all.py --no-dagster # Skip Dagster
    python scripts/run_all.py --no-mlflow  # Skip MLflow UI
    python scripts/run_all.py stop         # Stop all services from a previous run

Press Ctrl+C to stop all services.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PID_FILE = PROJECT_ROOT / ".run_all.pid"

SERVICES = {
    "api": {
        "cmd": [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", "8000", "--host", "127.0.0.1"],
        "url": "http://localhost:8000/health",
        "label": "FastAPI",
        "port": 8000,
    },
    "streamlit": {
        "cmd": [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                "--server.port", "8501", "--server.headless", "true"],
        "url": "http://localhost:8501",
        "label": "Streamlit Dashboard",
        "port": 8501,
    },
    "dagster": {
        "cmd": [sys.executable, "-m", "dagster", "dev", "-m", "ml.pipelines.definitions", "-p", "3000"],
        "url": "http://localhost:3000",
        "label": "Dagster",
        "port": 3000,
    },
    "mlflow": {
        "cmd": [sys.executable, "-m", "mlflow", "ui", "--port", "5000", "--backend-store-uri", "file:./mlruns"],
        "url": "http://localhost:5000",
        "label": "MLflow UI",
        "port": 5000,
    },
}

processes: dict[str, subprocess.Popen] = {}
shutting_down = False


def start_service(name: str) -> subprocess.Popen:
    svc = SERVICES[name]
    print(f"  Starting {svc['label']} -> {svc['url']}")
    proc = subprocess.Popen(
        svc["cmd"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def stop_all():
    global shutting_down
    if shutting_down:
        return
    shutting_down = True

    print("\nShutting down all services...")
    for name, proc in processes.items():
        label = SERVICES[name]["label"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
                print(f"  {label} stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print(f"  {label} killed")
        else:
            print(f"  {label} already exited")

    if PID_FILE.exists():
        PID_FILE.unlink()
    print("All services stopped.")


def stop_from_pid_file():
    """Stop services started by a previous run using the PID file."""
    if not PID_FILE.exists():
        print("No running services found (no .run_all.pid file).")
        return

    pids = PID_FILE.read_text().strip().splitlines()
    print(f"Stopping {len(pids)} processes from previous run...")

    for line in pids:
        name, pid_str = line.split("=", 1)
        pid = int(pid_str)
        label = SERVICES.get(name, {}).get("label", name)
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  {label} (PID {pid}) terminated")
        except (OSError, ProcessLookupError):
            print(f"  {label} (PID {pid}) already gone")

    PID_FILE.unlink()
    print("Done.")


def save_pid_file():
    lines = [f"{name}={proc.pid}" for name, proc in processes.items()]
    PID_FILE.write_text("\n".join(lines))


def check_health(name: str) -> bool:
    url = SERVICES[name]["url"]
    try:
        urllib.request.urlopen(url, timeout=2)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError):
        return False


def check_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def handle_signal(*_):
    stop_all()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Launch all AreteusML services")
    parser.add_argument("command", nargs="?", default="start", choices=["start", "stop"],
                        help="'start' (default) or 'stop' to kill a previous run")
    parser.add_argument("--no-dagster", action="store_true", help="Skip Dagster")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow UI")
    args = parser.parse_args()

    if args.command == "stop":
        stop_from_pid_file()
        return

    enabled = ["api", "streamlit"]
    if not args.no_dagster:
        enabled.append("dagster")
    if not args.no_mlflow:
        enabled.append("mlflow")

    # Check for port conflicts before starting
    blocked = []
    for name in enabled:
        port = SERVICES[name]["port"]
        if not check_port_free(port):
            blocked.append((name, port))
    if blocked:
        print("ERROR: Ports already in use:")
        for name, port in blocked:
            print(f"  :{port} ({SERVICES[name]['label']})")
        if PID_FILE.exists():
            print(f"\nTry: python scripts/run_all.py stop")
        else:
            print("\nFind and kill the processes using those ports.")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Clean up corrupted MLflow metrics from any previous interrupted runs
    from ml.utils.reproducibility import cleanup_mlflow_nulls
    cleaned = cleanup_mlflow_nulls(str(PROJECT_ROOT / "mlruns"))
    if cleaned:
        print(f"  Cleaned {cleaned} corrupted MLflow metric file(s) from a previous interrupted run\n")

    print("AreteusML - Starting services...\n")

    for name in enabled:
        processes[name] = start_service(name)

    save_pid_file()

    # Wait for services to come up
    print("\nWaiting for services to be ready...")
    time.sleep(8)

    all_ready = True
    for name in enabled:
        proc = processes[name]
        if proc.poll() is not None:
            print(f"  {SERVICES[name]['label']}: FAILED (exit code {proc.returncode})")
            all_ready = False
            continue
        healthy = check_health(name)
        status = "READY" if healthy else "starting..."
        print(f"  {SERVICES[name]['label']}: {status} -> {SERVICES[name]['url']}")

    if not all_ready:
        print("\nSome services failed. Check the logs. Press Ctrl+C to stop.")
    else:
        print("\nAll services running. Press Ctrl+C to stop.\n")

    # Keep alive, report crashes (no auto-restart to avoid zombie issues)
    try:
        while True:
            time.sleep(5)
            if shutting_down:
                break
            for name, proc in list(processes.items()):
                if proc.poll() is not None:
                    print(f"WARNING: {SERVICES[name]['label']} exited with code {proc.returncode}")
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
