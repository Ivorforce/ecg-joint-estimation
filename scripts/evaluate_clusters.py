"""How many beat clusters the fits used (paper, Method: "Clean 10-second
recordings typically reduce to one or two clusters; a second usually contains
a single beat").

    python scripts/evaluate_clusters.py
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

from common import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--fits", nargs="+", default=["fits/ludb", "fits/medalcare"])
ap.add_argument("--out", default="results")
args = ap.parse_args()

summary = {}
for fits in args.fits:
    n_clusters, minor_sizes = Counter(), Counter()
    for path in glob.glob(os.path.join(fits, "**", "*.npz"), recursive=True):
        sizes = np.bincount(np.load(path)["clusters"])
        n_clusters[len(sizes)] += 1
        # Every cluster but the largest.
        minor_sizes.update(np.sort(sizes)[:-1].tolist())
    summary[fits] = {"records_by_cluster_count": dict(sorted(n_clusters.items())),
                     "beats_in_minor_clusters": dict(sorted(minor_sizes.items()))}
    print(f"{fits}: records by cluster count {dict(sorted(n_clusters.items()))}, "
          f"minor cluster sizes {dict(sorted(minor_sizes.items()))}")
write_json(os.path.join(args.out, "clusters.json"), summary)
