"""E6: MiniGrid ablation smoke — active CTRO vs E0 locks (CPU)."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="E6 ablation smoke (sequential CPU runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cmds = [
        [sys.executable, "-m", "src.experiments.exp_e0_vanilla", "--smoke", "--seed", str(args.seed)],
        [sys.executable, "-m", "src.experiments.exp_e0_aa_ppo", "--smoke", "--seed", str(args.seed)],
        [sys.executable, "-m", "src.experiments.exp_active_ctro", "--smoke", "--seed", str(args.seed), "--device", "cpu"],
    ]
    for cmd in cmds:
        print("running", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("E6 ablation smoke complete")


if __name__ == "__main__":
    main()
