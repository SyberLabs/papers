import pandas as pd
import numpy as np
import glob
import os
import scipy.stats as stats

def test_backlog():
    # Find all the representational crossing traces for MLPs that successfully tracked rule signal
    # We'll stick to a fixed modulus (p=59) to isolate the latency/backlog effect driven by weight decay.
    # The traces vary by wd1 vs wd2, baseline vs spectral, etc.
    trace_files = glob.glob("d:/syberlabs/research/isomorphic/examples/representational_crossing_p59_mlp_wd*_*.csv")
    
    if not trace_files:
        print("No files found! Looking wider...")
        trace_files = glob.glob("d:/syberlabs/research/isomorphic/examples/representational_crossing_*_mlp_*.csv")
        
    results = []
    
    for f in trace_files:
        df = pd.read_csv(f)
        if "rule_signal" not in df.columns or "val_acc" not in df.columns:
            continue
            
        # Ensure it actually grokked (val acc hits high)
        if df["val_acc"].max() < 90.0:
            continue
            
        # The variables we want:
        # t_crit = grok_onset_epoch (the 50% mark of val acc)
        # But wait, there's a column 'grok_onset_epoch' appended to every row if it was summarized.
        if "grok_onset_epoch" in df.columns:
            t_crit = df["grok_onset_epoch"].iloc[0]
            if pd.isna(t_crit):
                continue
        else:
            # manually calculate
            final_val = df["val_acc"].iloc[-1]
            start_val = df["val_acc"].iloc[0]
            t_crit_idx = (df["val_acc"] >= start_val + 0.5 * (final_val - start_val)).idxmap()
            t_crit = df.loc[t_crit_idx, "epoch"]
            
        # Calculate Backlog Integral: Integral of R(t) dt up to t_crit
        # using trapezoidal rule or just simple sum since dt is constant (usually 20 epochs)
        pre_grok = df[df["epoch"] <= t_crit]
        epochs = pre_grok["epoch"].values
        dt = np.diff(epochs, prepend=0)
        # First dt is just epoch[0]
        dt[0] = epochs[0] 
        r_t = pre_grok["rule_signal"].values
        backlog_integral = np.sum(r_t * dt)
        
        # Calculate Magnitude of Deployment (Max Velocity of D)
        # We look around the grokking period
        peak_velocity = df["val_acc_velocity"].max()
        
        # We also look at the total duration of Phase 2
        latency_duration = t_crit
        
        results.append({
            "File": os.path.basename(f),
            "Latency (t_crit)": t_crit,
            "Backlog Integral (Int R dt)": float(backlog_integral),
            "Deployment Velocity (max dD/dt)": float(peak_velocity)
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Backlog Integral (Int R dt)")
    
    print("=========================================================================")
    print(" THE LATENT BACKLOG HYPOTHESIS TEST")
    print("=========================================================================")
    print(res_df.to_string(index=False))
    
    if len(res_df) > 1:
        corr, p_val = stats.pearsonr(res_df["Backlog Integral (Int R dt)"], res_df["Deployment Velocity (max dD/dt)"])
        print("-------------------------------------------------------------------------")
        print(f"Pearson Correlation (Backlog vs Spurt Velocity): {corr:.3f} (p-value: {p_val:.3e})")
        print("=========================================================================")
        if corr > 0.8:
            print("CONCLUSION: Hypothesis CONFIRMED. The magnitude of the emergence spurt ")
            print("is directly structurally determined by the geometric buildup of latent ")
            print("rule structure during the memorization-suppressed Phase 2.")
            
if __name__ == "__main__":
    test_backlog()
