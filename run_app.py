import subprocess
import sys
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run streamlit from project root
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])