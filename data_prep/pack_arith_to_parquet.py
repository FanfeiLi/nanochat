"""Convert synthetic_k5_arith.jsonl into pretrain-compatible parquet shards.

Each JSONL line is a list of messages like
  [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
We flatten the conversation to a single plain-text doc
  "<user>\\n<assistant>"
and write a parquet with columns [id: large_string, text: large_string] matching
the base shard schema (so the existing BOS-aligned dataloader can read it).

Row group size matches the other _rg2k parquets so 8-GPU per-file partitioning
still works cleanly.

Output:
  /fast/fli/synth_arith_parquets/shard_arith_00000.parquet
  /fast/fli/synth_arith_parquets/shard_arith_00001.parquet
  ...

Default: split into 5 shards (each ~10M tokens / ~300K conversations).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_IN  = "/fast/fli/synthetic_k5_arith.jsonl"
DEFAULT_OUT = "/fast/fli/synth_arith_parquets"
DEFAULT_NUM_SHARDS = 5
ROW_GROUP_SIZE = 2000


def flatten(messages: list[dict]) -> str:
    """User \\n Assistant — plain text, no chat markers."""
    return "\n".join(m.get("content", "") or "" for m in messages)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in",  dest="in_path",  default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_dir", default=DEFAULT_OUT)
    ap.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count lines (so we can balance shards)
    print(f"Counting lines in {in_path} ...")
    with open(in_path) as f:
        total = sum(1 for _ in f)
    per_shard = (total + args.num_shards - 1) // args.num_shards
    print(f"Total: {total:,} conversations; {args.num_shards} shards of ~{per_shard:,} each")

    # Second pass: write shards
    schema = pa.schema([
        pa.field("id",   pa.large_string()),
        pa.field("text", pa.large_string()),
    ])

    shard_idx = 0
    ids: list[str] = []
    texts: list[str] = []

    def flush(final: bool = False) -> None:
        nonlocal shard_idx, ids, texts
        if not ids:
            return
        out = out_dir / f"shard_arith_{shard_idx:05d}.parquet"
        table = pa.table({
            "id":   pa.array(ids,   type=pa.large_string()),
            "text": pa.array(texts, type=pa.large_string()),
        }, schema=schema)
        pq.write_table(table, out, row_group_size=ROW_GROUP_SIZE, compression="zstd")
        print(f"  wrote {out.name}  rows={len(ids):,}")
        shard_idx += 1
        ids.clear()
        texts.clear()

    with open(in_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)
            ids.append(f"arith_{i:07d}")
            texts.append(flatten(msgs))
            if len(ids) >= per_shard:
                flush()
    flush(final=True)

    print(f"\nDone. {shard_idx} shards written to {out_dir}/")
    for p in sorted(out_dir.glob("shard_arith_*.parquet")):
        sz = p.stat().st_size / 1e6
        print(f"  {p.name:30s}  {sz:6.1f} MB")


if __name__ == "__main__":
    main()
