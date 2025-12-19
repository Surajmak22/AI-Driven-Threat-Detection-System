#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
from pathlib import Path


def find_available_port(start_port=8501, max_attempts=20):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def main():
    # Add src directory to Python path
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Get the dashboard path
    dashboard_path = str(Path(__file__).parent / "src" / "dashboard" / "phase1_dashboard.py")
    
    # Find an available port
    port = find_available_port(8501)
    
    # Build the Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.serverAddress", "localhost",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        dashboard_path
    ]
    
    # Print connection info
    print(f"\nStarting SmartGuard AI Phase-1 Dashboard...")
    print(f"• Local:   http://localhost:{port}")
    print("• Press Ctrl+C to stop\n")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
