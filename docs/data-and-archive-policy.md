# Data and archive policy

## Archive directory (`_archive/`)

Material under `_archive/` is kept as a safety net during restructuring. Those paths are git-ignored and are **not** part of the active import graph. Nothing there is required to run the current scraper, dataset construction, training, or inference paths documented in the root README.

## Non-destructive workflows

Pipeline steps should **not** delete raw dataset assets as a routine way to fix or clean data. Prefer new manifest versions under `src/dataset_construction/manifests/`, reports, or metadata flags so provenance stays recoverable.

The repository automation rule is documented in [`.cursor/rules/never-delete-data.mdc`](../.cursor/rules/never-delete-data.mdc): avoid destructive removal of raw dataset or user-generated assets unless you explicitly intend it; prefer manifests, flags, or moving material to dated or archive folders.

## Heavy artifacts

Training videos, weights, and large exports live under `data/` and `models/` on your machine. See [data/README.md](../data/README.md), [models/README.md](../models/README.md), and [src/inference/README.md](../src/inference/README.md) for expected paths.
