#!/usr/bin/env bash
# Build the private Kaggle dataset that carries this repo and ml-ladder into kernels.
# The repos are private, so kernels cannot clone them; ml-ladder delivered its source
# the same way (CLAUDE.md, "Getting the code in"). Committed HEADs only (git archive),
# so what runs on Kaggle is exactly a commit.
#
#   tools/kaggle/package_source.sh          # writes runs/kaggle-source/
#   kaggle datasets create -p runs/kaggle-source --dir-mode skip      # first time (private)
#   kaggle datasets version -p runs/kaggle-source -m "<commit>"       # afterwards
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
ladder="${ML_LADDER:-$here/../ml-ladder}"
out="$here/runs/kaggle-source"
rm -rf "$out" && mkdir -p "$out"
git -C "$here" archive --format=tar --prefix=jax-experiments/ HEAD > "$out/jax-experiments.tar"
git -C "$ladder" archive --format=tar --prefix=ml-ladder/ HEAD > "$out/ml-ladder.tar"
cat > "$out/dataset-metadata.json" <<JSON
{"title": "jaxexp-source", "id": "davidmiheev/jaxexp-source", "licenses": [{"name": "other"}]}
JSON
printf 'jax-experiments %s\nml-ladder %s\n' "$(git -C "$here" rev-parse HEAD)" "$(git -C "$ladder" rev-parse HEAD)" > "$out/COMMITS"
ls -la "$out"
