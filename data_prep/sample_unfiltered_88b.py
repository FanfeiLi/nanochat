"""Stratified-sample whole parquet files from fineweb_edu_score_2 (unfiltered)
until the accumulated token_count reaches --target-tokens.

Strategy:
  - Enumerate all CC-MAIN-* dump subdirs under --data-root.
  - For each dump, shuffle its file list with a fixed seed.
  - Interleave by round: round r takes file[r] from each dump (dumps shuffled too).
    This gives ~equal coverage across all dumps before revisiting any one twice.
  - For each chosen file, write a slim copy with only ['id', 'text'] columns,
    row_group_size=1000, to --output as shard_XXXXX.parquet.
  - Stop as soon as accumulated token_count >= --target-tokens.
  - Write MANIFEST.csv mapping output shard -> source file -> tokens.

Output is drop-in for nanochat speedrun_cluster_d34_newtok.sh: point
NEW_DATASET at --output. Last shard is used as val by dataset.parquets_iter_batched.
"""
import argparse
import os
import random
import time
import multiprocessing as mp
import pyarrow.parquet as pq


def build_ordered_list(data_root: str, seed: int):
    dumps = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    rng = random.Random(seed)
    rng.shuffle(dumps)

    per_dump_files = {}
    for d in dumps:
        files = sorted(
            f for f in os.listdir(os.path.join(data_root, d))
            if f.endswith(".parquet")
        )
        rng.shuffle(files)
        per_dump_files[d] = files

    max_rounds = max(len(v) for v in per_dump_files.values())
    ordered = []
    for r in range(max_rounds):
        for d in dumps:
            files = per_dump_files[d]
            if r < len(files):
                ordered.append(os.path.join(data_root, d, files[r]))
    return ordered


def process_one(args):
    src, dst = args
    if os.path.exists(dst):
        # already done — recompute tokens from source for manifest
        t = pq.read_table(src, columns=["token_count"])
        tokens = int(t.column("token_count").to_numpy().sum())
        return (dst, src, tokens, True)

    t = pq.read_table(src, columns=["id", "text", "token_count"])
    tokens = int(t.column("token_count").to_numpy().sum())
    slim = t.select(["id", "text"])
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(slim, tmp, row_group_size=1000)
    os.rename(tmp, dst)
    return (dst, src, tokens, False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/scratch/fli/fineweb_data/fineweb_edu_score_2/data")
    p.add_argument("--output", default="/fast/fli/base_data_fineweb_unfiltered_rg2k")
    p.add_argument("--target-tokens", type=int, default=88_000_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ordered = build_ordered_list(args.data_root, args.seed)
    print(f"Enumerated {len(ordered)} candidate files across all dumps")
    print(f"Target: {args.target_tokens/1e9:.1f}B tokens -> {args.output}")

    manifest_path = os.path.join(args.output, "MANIFEST.csv")
    mf = open(manifest_path, "w")
    mf.write("shard,source,tokens\n")

    total_tokens = 0
    shards_written = 0
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        i = 0
        while total_tokens < args.target_tokens and i < len(ordered):
            batch_end = min(i + args.batch_size, len(ordered))
            work = [
                (ordered[j], os.path.join(args.output, f"shard_{j:05d}.parquet"))
                for j in range(i, batch_end)
            ]
            for dst, src, tokens, was_skipped in pool.imap_unordered(process_one, work):
                total_tokens += tokens
                shards_written += 1
                mf.write(f"{os.path.basename(dst)},{src},{tokens}\n")
                mf.flush()
            elapsed = time.time() - t0
            print(
                f"[{batch_end}/{len(ordered)}] shards={shards_written} "
                f"tokens={total_tokens/1e9:.2f}B / {args.target_tokens/1e9:.1f}B "
                f"elapsed={elapsed/60:.1f}min",
                flush=True,
            )
            i = batch_end

    mf.close()
    print(
        f"Done. {shards_written} shards, {total_tokens/1e9:.3f}B tokens, "
        f"{(time.time()-t0)/60:.1f} min -> {args.output}"
    )


if __name__ == "__main__":
    main()
