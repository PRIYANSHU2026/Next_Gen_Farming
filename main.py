#!/usr/bin/env python3
"""
Entry point to run the Next-Gen Farming Streamlit dashboard.
This launches the app defined in `src/simple_esp32_dashboard.py`.
Run with: `python3 main.py`
"""

import os
import sys
import subprocess


def run_streamlit_dashboard():
    project_root = os.path.dirname(__file__)
    src_dir = os.path.join(project_root, "src")
    script_path = os.path.join(src_dir, "simple_esp32_dashboard.py")

    if not os.path.exists(script_path):
        print(f"Error: dashboard script not found at {script_path}")
        sys.exit(1)

    # Build the command to run Streamlit via the current Python interpreter
    cmd = [sys.executable, "-m", "streamlit", "run", script_path]

    # Forward any additional CLI args (e.g., port or headless) to Streamlit
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Set working directory to src so relative paths (like crop_recommendations.csv) resolve
    subprocess.run(cmd, cwd=src_dir)


if __name__ == "__main__":
    run_streamlit_dashboard()