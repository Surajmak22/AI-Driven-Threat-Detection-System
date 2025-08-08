#!/usr/bin/env python3
"""
SmartGuard AI - Dashboard Launcher

This script launches the Streamlit dashboard for the SmartGuard AI system.
"""
import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required Python packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    """Main entry point for the dashboard launcher."""
    # Add the src directory to the Python path
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Check if requirements are installed
    try:
        import streamlit
        import plotly
    except ImportError:
        print("Required packages not found. Installing...")
        install_requirements()
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = "8502"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    print("Starting SmartGuard AI Dashboard...")
    print("Press Ctrl+C to stop the dashboard")
    print("Open http://localhost:8502 in your web browser")
    
    try:
        # Initialize and run the dashboard directly
        dashboard = SmartGuardDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
