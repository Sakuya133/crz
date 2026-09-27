#!/usr/bin/env python3
"""Streaming, resumable full-image embedding of an audited enrollment snapshot."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
import numpy as np
import pandas as pd
from PIL import Image
from cxr_steganalysis.config import load_config
from cxr_steganalysis.experiment import effective_seeds
from cxr_steganalysis.data.manifest import build_pair_manifest
from cxr_steganalysis.stego.lsb_matching import embed_lsb_matching_file, load_grayscale_png
from cxr_steganalysis.provenance import sha256_file


def prepare(config, preflight, output, max_pairs=None):
    preflight, output = Path(preflight), Path(output)
    split_path = preflight / "splits.csv"
    summary = json.loads((preflight/"summary.json").read_text())
    if config.get("dataset_profile") == "full_all_eligible" and (summary.get("missing_images", 0) or summary.get("exclusions", {}).get("unreadable_requires_retrieval", 0)):
        raise ValueError("Full NIH inventory is incomplete; refusing an implicit available-only subset")
    if sha256_file(split_path) != summary["split_identity"]:
        raise ValueError("Enrollment snapshot changed after preflight")
    split = pd.read_csv(split_path, dtype={"patient_id": str})
    payload = float(config["payload_bpp"])
    seed = effective_seeds(config)["embedding"]
    binding = dict(split_sha256=sha256_file(split_path), payload_bpp=payload, embedding_seed=seed, algorithm="lsb_matching", protocol_id=config["protocol_id"])
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output/"binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Preparation binding changed; use a separate payload/protocol directory")
    binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    records_dir, stego_dir = output/"records", output/"stego"
    records_dir.mkdir(exist_ok=True)
    stego_dir.mkdir(exist_ok=True)
    remaining = split[~split.image_id.map(lambda x: (records_dir/(x+".json")).exists())]
    # Conservative uncompressed estimate plus 10%; PNG can be much smaller.
    estimate = int((remaining.width*remaining.height).sum()*1.1)
    if estimate > shutil.disk_usage(output).free:
        raise OSError(f"Insufficient disk: conservative new stego estimate {estimate/2**30:.1f} GiB; no silent subset or payload duplication")
    records = []
    selected = split if max_pairs is None else split.head(max_pairs)
    for i, r in enumerate(selected.itertuples(index=False)):
        record_path = records_dir/(r.image_id+".json")
        target = stego_dir/r.image_id
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["cover_sha256"] != sha256_file(r.cover_path) or record["stego_sha256"] != sha256_file(target):
                raise ValueError(f"Cached preparation checksum failed: {r.image_id}")
        else:
            result = embed_lsb_matching_file(r.cover_path, payload, seed, r.image_id)
            if target.exists():
                if not np.array_equal(load_grayscale_png(target), result.image):
                    raise ValueError(f"Orphan stego differs from regeneration: {target}")
            else:
                tmp = target.with_suffix(".tmp.png")
                Image.fromarray(result.image).save(tmp)
                tmp.replace(target)
            record = dict(image_id=r.image_id, stego_path=str(target.resolve()), **result.metadata.to_dict())
            record.update(cover_sha256=sha256_file(r.cover_path), stego_sha256=sha256_file(target))
            tmp = record_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2)+"\n")
            tmp.replace(record_path)
        records.append(record)
        if (i+1) % 100 == 0:
            print(f"Prepared {i+1}/{len(selected)}; complete full-image LSB Matching, then crop at loading", flush=True)
    if max_pairs is not None and len(selected) != len(split):
        print(f"DEBUG PARTIAL ONLY: {len(selected)}/{len(split)}; no full manifest emitted")
        return
    generation = pd.DataFrame(records)
    generation.to_csv(output/"stego_generation.csv", index=False)
    pairs = build_pair_manifest(split, generation)
    if "test_cohort" in split:
        pairs = pairs.merge(split[["image_id", "test_cohort"]], on="image_id", validate="one_to_one")
    pairs.to_csv(output/"cover_stego.csv", index=False)
    (output/"complete.json").write_text(json.dumps(dict(binding=binding, pairs=len(pairs), manifest_sha256=sha256_file(output/"cover_stego.csv")), indent=2)+"\n")
    print(f"Prepared all {len(pairs)} available eligible pairs: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--debug-max-pairs", type=int, help="Explicit partial preparation diagnostic; never emits a full manifest")
    a = p.parse_args()
    prepare(load_config(a.config), a.preflight, a.output, a.debug_max_pairs)
