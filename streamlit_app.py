"""
Streamlit Cloud entry point for SmartGuard AI Dashboard
"""
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the dashboard
from src.dashboard.dashboard import SmartGuardDashboard

# Set page config
import streamlit as st
st.set_page_config(
    page_title="SmartGuard AI - Network Threat Detection",
    layout="wide",
    page_icon="🛡️"
)

# Initialize and run the dashboard
dashboard = SmartGuardDashboard()
dashboard.run()
