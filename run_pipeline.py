import subprocess
import sys

scripts = [
    "01_fetch_data.py",
    "02_build_returns.py",
    "03_replicate_ff.py",
    "04_build_signal.py",
    "05_ic_analysis.py",
    "06_walk_forward.py"
]

for script in scripts:
    subprocess.run([sys.executable, script], check=True)
