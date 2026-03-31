# gorse-optimize

Standalone binary for on-demand hyperparameter optimization of Gorse recommendation models. Runs independently of the normal pipeline cycle — no need to enable `optimize_period` or wait for a full training run.

## Usage

```bash
go build ./cmd/gorse-optimize

./gorse-optimize run --config /path/to/config.toml
./gorse-optimize list --cache-dir /path/to/gorse/cache
./gorse-optimize apply --cache-dir /path/to/gorse/cache
```

## Subcommands

### `run` — Run optimization

Loads data, runs goptuna TPE optimization for CF (BPR/ALS) and CTR (AFM) models, saves results to SQLite.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to gorse TOML config |
| `--trials` | 10 | Number of goptuna trials per model |
| `--jobs` | NumCPU | Parallel workers for model fitting |
| `--patience` | 10 | Early stopping patience |
| `--split-ratio` | 0.2 | Fraction of CTR data used for testing (0.0-1.0) |
| `--quiet` | false | Suppress log output |
| `--log-path` | | Path to log file |

Logging is on by default with console-formatted output. Use `--quiet` to suppress.

### `list` — List runs and current meta store values

Shows the current model parameters in the gorse meta store (with timestamps), followed by a reverse-chronological list of optimization runs.

| Flag | Default | Description |
|------|---------|-------------|
| `--limit` | 20 | Max number of runs to show |

### `apply` — Apply results to the gorse meta store

Writes optimization results into the gorse meta store (`meta.sqlite3`), making them the active parameters for the next model fit.

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | 0 | Specific run ID to apply (default: latest of each model type) |

## Global flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `./optimize-results.sqlite3` | Path for results SQLite database |
| `--cache-dir` | `os.TempDir()` | Gorse cache directory containing `meta.sqlite3` |

## Output

Results are stored in the `runs` table:

| Column | Description |
|--------|-------------|
| `model_type` | `MF` or `FM` |
| `best_type` | Best model variant (e.g. `BPR`, `ALS`, `FM`) |
| `params` | JSON hyperparameters |
| `score` | JSON scores (NDCG/Precision/Recall for MF, AUC/Precision/Recall for FM) |
| `trials` | Number of trials run |
| `duration_seconds` | Wall time for optimization |
| `created_at` | Timestamp |

## Tuning tips

The CTR (AFM) model dominates runtime. These are the main levers for speed vs. effectiveness:

| Setting | Default | Recommendation | Impact |
|---------|---------|----------------|--------|
| `--patience` | 10 | 5 | AFM scores plateau early; patience 5 catches it faster, saving many epochs per trial. Biggest win. |
| `--split-ratio` | 0.2 | 0.1 | Smaller test set means faster evaluation per epoch. Minor accuracy trade-off. |
| `--trials` | 10 | 8-12 | Each CTR trial takes 25-50 min. Fewer trials = proportionally faster. |
| `fit_epoch` (config) | 100 (BPR), 50 (ALS/AFM) | Keep defaults | Early stopping usually kicks in well before the epoch limit. |
| `optimize_trials` (config) | 10 | Overridden by `--trials` | CLI flag takes precedence. |

## Examples

Quick test with 2 trials:

```bash
./gorse-optimize run --config config.toml --trials 2
```

Fast run with tuned settings:

```bash
./gorse-optimize run --config config.toml --trials 8 --patience 5 --split-ratio 0.1
```

List runs and current meta store values:

```bash
./gorse-optimize list --cache-dir ~/dev/nrc/gorse-local/var-lib-gorse
```

Apply latest optimization results:

```bash
./gorse-optimize apply --cache-dir ~/dev/nrc/gorse-local/var-lib-gorse
```

Apply a specific run by ID:

```bash
./gorse-optimize apply --cache-dir ~/dev/nrc/gorse-local/var-lib-gorse --run-id 3
```
