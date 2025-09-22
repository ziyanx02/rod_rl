#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Continuously rerun run_cmaes.py with --task.")
    ap.add_argument("--task", type=str, required=True, help="Task to pass to run_cmaes.py")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait before restarting after exit (default: 2.0)")
    args = ap.parse_args()

    cmd = ["python", "run_cmaes.py", "--task", args.task]

    print(f"[supervisor] Starting loop. Will run: {' '.join(cmd)}")
    try:
        i = 1
        while True:
            print(f"\n[supervisor] Launch #{i} at {datetime.now().isoformat(timespec='seconds')}")
            # Run the child; do not raise on non-zero (we want to restart regardless)
            result = subprocess.run(cmd)
            print(f"[supervisor] Child exited with return code {result.returncode}")
            i += 1
            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n[supervisor] Stopped by user. Bye!")

if __name__ == "__main__":
    main()
