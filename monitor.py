import time
import os
import json
import subprocess
import sys

run_dir = "outputs/exp01_baseline/metrics/run_20260306_111302"
compose_file = "docker/docker-compose.exp01.yml"

print("Starting monitor for round 15 checkpoint...", flush=True)

round_15_checked = False

while True:
    rounds = []
    if os.path.exists(run_dir):
        rounds = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(run_dir) if f.startswith('round_') and f.endswith('.json')]
    
    current = max(rounds) if rounds else 0
    print(f"Current round finished: {current}", flush=True)
    
    if current >= 15 and not round_15_checked:
        round_15_file = os.path.join(run_dir, "round_15.json")
        try:
            with open(round_15_file, "r") as f:
                data = json.load(f)
            acc = data.get("global_accuracy", 0.0)
            print(f"ROUND 15 CHECKPOINT! global_accuracy = {acc:.4f}", flush=True)
            if acc < 0.35:
                print("Accuracy below 35%. Aborting experiment.", flush=True)
                subprocess.run(["docker", "compose", "-f", compose_file, "down"])
                sys.exit(1)
            else:
                print("Checkpoint PASSED. Continuing to 50 rounds.", flush=True)
                round_15_checked = True
        except Exception as e:
            print(f"Failed to read round 15: {e}", flush=True)
            
    if current >= 50:
        print("Experiment 50 rounds completed successfully.", flush=True)
        break
        
    res = subprocess.run(["docker", "ps", "-q", "-f", "name=fl_server_exp01"], capture_output=True, text=True)
    if not res.stdout.strip() and current > 0:
        print("Server container not found. It might have finished or crashed.", flush=True)
        break

    time.sleep(30)
