"""Build the mixed mid-training corpus directory.

Interleaves base shards (hardlinks) and synthetic arithmetic shards (hardlinks)
into a single dir so the nanochat dataloader — which sorts filenames
alphabetically — reads them in interleaved order. This keeps arithmetic
exposure spread across training, not clumped at the end.

Layout pattern (default 4:1 base:arith, 8 base shards + 2 arith shards):
  /fast/fli/midtrain_corpus_k5arith/
  ├── shard_000_base.parquet     (hardlink → /fast/fli/base_data_passed_both_d4_rg2k_symclean/shard_02315.parquet)
  ├── shard_001_base.parquet     (hardlink)
  ├── shard_002_base.parquet     (hardlink)
  ├── shard_003_base.parquet     (hardlink)
  ├── shard_004_arith.parquet    (hardlink)
  ├── shard_005_base.parquet
  ...
  └── shard_val.parquet          (last file = val split)

Python `sorted()` order is lexicographic, so the 3-digit zero-padded prefix
gives exact interleaved order regardless of "_base"/"_arith" suffix.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_BASE_DIR    = Path("/fast/fli/base_data_passed_both_d4_rg2k_symclean")
DEFAULT_ARITH_DIR   = Path("/fast/fli/synth_arith_parquets")
DEFAULT_OUT_DIR     = Path("/fast/fli/midtrain_corpus_k5arith")

# Which unused base shards to replay. Shards 02315..02548 are untouched by
# d34_newtok training (see /fast/fli/d34_newtok_unused_shards.txt).
DEFAULT_BASE_START_IDX = 2315
DEFAULT_NUM_BASE_SHARDS = 8   # ~275M base tokens
DEFAULT_RATIO_BASE_PER_ARITH = 4   # 4 base shards per 1 arith shard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir",   type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--arith-dir",  type=Path, default=DEFAULT_ARITH_DIR)
    ap.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--base-start-idx",       type=int, default=DEFAULT_BASE_START_IDX,
                    help="first unused base shard index (e.g. 2315)")
    ap.add_argument("--num-base-shards",      type=int, default=DEFAULT_NUM_BASE_SHARDS,
                    help="how many unused base shards to pull in")
    ap.add_argument("--ratio-base-per-arith", type=int, default=DEFAULT_RATIO_BASE_PER_ARITH,
                    help="base:arith ratio, interleaved. e.g. 4 = 1 arith every 4 base")
    ap.add_argument("--val-source", choices=["base", "arith"], default="arith",
                    help="source for the final val shard (split off the end)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale entries (user may re-run with different ratios)
    for old in args.out_dir.glob("shard_*.parquet"):
        old.unlink()

    # Source lists
    base_shards = sorted(args.base_dir.glob("shard_*.parquet"))
    # filter to those at/after --base-start-idx
    def shard_num(p: Path) -> int:
        stem = p.stem  # shard_02315
        return int(stem.split("_")[1])
    base_candidates = [p for p in base_shards if shard_num(p) >= args.base_start_idx]
    if len(base_candidates) < args.num_base_shards:
        raise RuntimeError(f"Only {len(base_candidates)} unused base shards available "
                           f"from idx {args.base_start_idx}; requested {args.num_base_shards}")
    chosen_base = base_candidates[:args.num_base_shards]

    arith_shards = sorted(args.arith_dir.glob("shard_arith_*.parquet"))
    if not arith_shards:
        raise RuntimeError(f"No arith shards found in {args.arith_dir}. "
                           "Run pack_arith_to_parquet.py first.")

    # Interleave: every `ratio_base_per_arith` base shards followed by 1 arith.
    # We'll consume arith shards cyclically if the ratio math produces more arith
    # slots than available arith shards; otherwise just walk them in order.
    interleaved: list[tuple[str, Path]] = []  # (role, source_path)
    base_iter = iter(chosen_base)
    arith_iter = iter(arith_shards)
    base_in_chunk = 0
    while True:
        try:
            b = next(base_iter)
        except StopIteration:
            break
        interleaved.append(("base", b))
        base_in_chunk += 1
        if base_in_chunk >= args.ratio_base_per_arith:
            try:
                a = next(arith_iter)
                interleaved.append(("arith", a))
            except StopIteration:
                pass  # ran out of arith — just keep going with base
            base_in_chunk = 0

    # If we have arith shards left over, tail them on (helps if you want 100% of arith)
    for a in arith_iter:
        interleaved.append(("arith", a))

    # Decide which one becomes the val shard (the LAST entry, which the
    # dataloader peels off as val). Prefer a small clean arith or base shard.
    # Simple policy: take the last entry of type specified by --val-source.
    val_entry = None
    for role, p in reversed(interleaved):
        if role == args.val_source:
            val_entry = (role, p)
            break
    if val_entry is None:
        val_entry = interleaved[-1]
    interleaved_remaining = [e for e in interleaved if e is not val_entry]

    # Hardlink into output dir with alphabetically ordered names
    total_size = 0
    for i, (role, src) in enumerate(interleaved_remaining):
        dst = args.out_dir / f"shard_{i:04d}_{role}.parquet"
        os.link(src, dst)
        total_size += dst.stat().st_size

    val_role, val_src = val_entry
    val_dst = args.out_dir / f"shard_val_{val_role}.parquet"  # last alphabetically (starts with "shard_val_")
    # NOTE: "shard_val_base.parquet" sorts AFTER "shard_9999_*.parquet" if we had that
    # many, which we don't — so it's the alphabetical last.
    os.link(val_src, val_dst)
    total_size += val_dst.stat().st_size

    # Summary
    print(f"\n=== corpus built at {args.out_dir} ===")
    print(f"  ratio base:arith = {args.ratio_base_per_arith}:1")
    print(f"  train shards     = {len(interleaved_remaining)}")
    print(f"    base           = {sum(1 for r, _ in interleaved_remaining if r == 'base')}")
    print(f"    arith          = {sum(1 for r, _ in interleaved_remaining if r == 'arith')}")
    print(f"  val shard        = {val_dst.name} ({val_role})")
    print(f"  total size       = {total_size / 1e9:.1f} GB")
    # Rough token estimate: 34M tok/base shard, 10M tok/arith shard
    base_toks  = sum(1 for r, _ in interleaved_remaining if r == "base") * 34_000_000
    arith_toks = sum(1 for r, _ in interleaved_remaining if r == "arith") * 10_000_000
    total_toks = base_toks + arith_toks
    arith_pct  = 100.0 * arith_toks / total_toks if total_toks else 0.0
    print(f"  est. train tokens ≈ {total_toks / 1e6:.0f} M  "
          f"(base {base_toks/1e6:.0f}M + arith {arith_toks/1e6:.0f}M = {arith_pct:.1f}% arith)")

    print(f"\nNext:")
    print(f"  1. point NANOCHAT_BASE_DIR/base_data_climbmix symlink at {args.out_dir}")
    print(f"  2. launch mid_train_text with --init-from-ckpt-dir=/fast/fli/.cache/nanochat_d34_newtok/base_checkpoints/d34 --init-from-step=38207")


if __name__ == "__main__":
    main()
