"""Rewrite K5 parquets with multiple row groups for nanochat multi-GPU compatibility.

Usage:
    python rewrite_rowgroups.py --input /fast/fli/base_data_k5 --output /fast/fli/base_data_k5_rg --row-group-size 10000
    python rewrite_rowgroups.py --input /fast/fli/base_data_k5 --output /fast/fli/base_data_k5_rg --row-group-size 10000 --max-files 5  # quick test
"""
import argparse
import os
import time
import multiprocessing as mp
import pyarrow.parquet as pq


def process_file(args):
    src, dst, row_group_size = args
    if os.path.exists(dst):
        return "skipped"
    table = pq.read_table(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(table, dst, row_group_size=row_group_size)
    return f"{table.num_rows} rows"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--row-group-size", type=int, default=10000)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input) if f.endswith(".parquet"))
    if args.max_files:
        files = files[:args.max_files]

    workers = args.workers or mp.cpu_count()
    work = [(os.path.join(args.input, f), os.path.join(args.output, f), args.row_group_size) for f in files]

    already = sum(1 for _, dst, _ in work if os.path.exists(dst))
    print(f"Total: {len(files)}, already done: {already}, remaining: {len(files) - already}, workers: {workers}")

    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool:
        for result in pool.imap_unordered(process_file, work):
            done += 1
            if done % 100 == 0 or done == len(files):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(files)}] {rate:.1f} files/s  ETA {eta/60:.1f}min")

    print(f"Done. {len(files)} files in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
