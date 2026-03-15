import os
import subprocess
import sys

def run_step(command):
    print(f"Running: {command}")
    # Replace 'python' with the path to the current interpreter, wrapping in quotes for spaces
    if command.startswith("python "):
        command = f'"{sys.executable}" ' + command[7:]
    
    # Set PYTHONPATH to include the 'src' directory to allow cross-module imports
    env = os.environ.copy()
    src_path = os.path.join(os.getcwd(), "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    
    subprocess.run(command, shell=True, check=True, env=env)

if __name__ == "__main__":
    # 1. Download & Preprocess (Small data: 100 cells, 100 genes)
    run_step("python src/data/preprocess.py --n_cells 100 --n_genes 100")
    
    # 2. Trajectory
    run_step("python src/utils/trajectory.py")
    
    # 3. Features & Priors
    run_step("python src/data/features.py")
    
    # 4. Train (Only 2 epochs for speed)
    run_step("python src/train.py --epochs 2")
    
    # 5. Visualize
    run_step("python src/utils/visualization.py")
    
    print("Pipeline finished. To view the dashboard, run: ./venv/bin/streamlit run scripts/dashboard.py")
