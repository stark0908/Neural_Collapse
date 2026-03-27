import os
import itertools
import pandas as pd
import numpy as np
import multiprocessing
import multiprocessing.pool
import argparse

from train_clip import train_model, fractions, domains, methods, OUT_FILE

def get_combinations():
    regimes = []
    # 3→1
    for test in domains:
        train = [d for d in domains if d != test]
        regimes.append((train, [test]))
    # 2→2
    for train_pair in itertools.combinations(domains, 2):
        test_pair = [d for d in domains if d not in train_pair]
        regimes.append((list(train_pair), test_pair))
    # 1→3
    for train in domains:
        test = [d for d in domains if d != train]
        regimes.append(([train], test))
        
    tasks = []
    for train_domains, test_domains in regimes:
        for frac in fractions:
            for cfg in methods:
                tasks.append((train_domains, test_domains, frac, cfg))
    return tasks

def worker_fn(task_data):
    # Unpack task
    task, lock = task_data
    train_domains, test_domains, frac, cfg = task
    
    # Run training
    avg_acc, worst_acc, train_loss, val_loss = train_model(train_domains, test_domains, frac, cfg)
    
    # Save safely
    with lock:
        if os.path.exists(OUT_FILE):
            df_upd = pd.read_csv(OUT_FILE)
        else:
            df_upd = pd.DataFrame(columns=[
                "train_domains", "test_domains", 
                "fraction", "method", 
                "avg_acc", "worst_acc",
                "train_loss", "val_loss"
            ])
            
        df_upd.loc[len(df_upd)] = [
            str(train_domains),
            str(test_domains),
            frac,
            cfg["name"],
            avg_acc,
            worst_acc,
            train_loss,
            val_loss
        ]
        tmp_file = OUT_FILE + ".tmp"
        df_upd.to_csv(tmp_file, index=False)
        os.replace(tmp_file, OUT_FILE)
        print(f"[SAVED] {train_domains} -> {test_domains} | Frac: {frac} | Method: {cfg['name']}")

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    # we can optionally pass gpu down to train_clip globals since they check sys.argv
    args, _ = parser.parse_known_args()

    tasks = get_combinations()

    # Filter out already completed tasks without lock (read-only initialization)
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"[INFO] Found existing results, {len(df)} runs completed.")
    else:
        df = pd.DataFrame(columns=[
            "train_domains", "test_domains", 
            "fraction", "method", 
            "avg_acc", "worst_acc",
            "train_loss", "val_loss"
        ])
        
    pending_tasks = []
    for t in tasks:
        train_domains, test_domains, frac, cfg = t
        mask = (
            (df["train_domains"] == str(train_domains)) &
            (df["test_domains"] == str(test_domains)) &
            (np.isclose(df["fraction"], frac)) &
            (df["method"] == cfg["name"])
        )
        if not mask.any():
            pending_tasks.append(t)

    print(f"[INFO] {len(pending_tasks)} tasks remaining out of {len(tasks)}.")

    if len(pending_tasks) == 0:
        print("All experiments completed!")
        return

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # Bundle task and lock
    pool_tasks = [(t, lock) for t in pending_tasks]

    # Use spawn to prevent CUDA fork issues
    ctx = multiprocessing.get_context("spawn")
    
    with ctx.Pool(processes=args.workers) as pool:
        # Use map to block until all complete while keeping workers busy
        pool.map(worker_fn, pool_tasks)
        
    print("[INFO] All parallel evaluations finished successfully.")

if __name__ == "__main__":
    main()
