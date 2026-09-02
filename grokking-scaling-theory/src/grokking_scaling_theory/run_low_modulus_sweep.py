import subprocess
import pandas as pd
from pathlib import Path
import sys

def main():
    target_primes = [31, 37, 41, 43, 47, 53]
    csv_out_dir = Path("d:/syberlabs/research/grokking-scaling-theory/data/")
    results = []

    print("=========================================================")
    print(" LAUNCHING LOW-MODULUS SWEEP ")
    print("=========================================================")
    
    script_path = "d:/syberlabs/research/isomorphic/examples/generate_grokking.py"

    for p in target_primes:
        print(f"--> Running p={p}...")
        csv_file = csv_out_dir / f"sweep_p{p}_raw.csv"
        
        cmd = [
            sys.executable,
            script_path,
            "--modulus", str(p),
            "--min-epochs", "20",   # Allow tiny models to grok extremely fast
            "--max-epochs", "20000",
            "--output", str(csv_file)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Read the resulting CSV to find the true grok epoch
        df = pd.read_csv(csv_file)
        grok_epoch = df["epoch"].iloc[-1]
        
        print(f"    Grokking completed at epoch: {grok_epoch}")
        results.append({
            "task_name": "modular_addition",
            "dataset_name": "modular_addition",
            "modulus": p,
            "weight_decay": 1.0,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "architecture": "mlp",
            "width": "",
            "depth": "",
            "noise_level": 0.0,
            "seed": 42,
            "max_epochs": 20000,
            "grokking_threshold": 95,
            "grokking_epoch": int(grok_epoch),
            "is_censored": "False",
            "include_in_scaling_fit": "True",
            "source": "local_sweep",
            "notes": "Low-modulus automated sweep point",
            "trace_path": str(csv_file)
        })
        
    df_results = pd.DataFrame(results)
    res_path = csv_out_dir / "low_modulus_sweep_results.csv"
    df_results.to_csv(res_path, index=False)
    print("=========================================================")
    print(f" DONE. Results written to {res_path}")

if __name__ == "__main__":
    main()
