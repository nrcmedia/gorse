# gorse-optimize

Standalone binary for on-demand hyperparameter optimization of Gorse recommendation models. Runs independently of the normal pipeline cycle — no need to enable `optimize_period` or wait for a full training run.

## Usage

```bash
go build ./cmd/gorse-optimize

./gorse-optimize --config /path/to/config.toml
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to gorse TOML config |
| `--trials` | 20 | Number of goptuna trials per model |
| `--jobs` | NumCPU | Parallel workers for model fitting |
| `--patience` | 10 | Early stopping patience |
| `--output` | `./optimize-results.sqlite3` | Path for output SQLite database |
| `--quiet` | false | Suppress log output |
| `--log-path` | | Path to log file |

## What it does

1. Loads the full dataset from the database configured in the TOML config
2. Splits the data:
   - Collaborative filtering: leave-one-out split
   - Click-through rate: 80/20 random split
3. Runs goptuna TPE optimization:
   - **MF**: searches over BPR and ALS models
   - **FM**: searches over AFM model
4. Logs progress per trial (trial number, score, best-so-far, duration)
5. Saves results to a local SQLite database
6. Prints a human-readable summary table to stdout

Logging is on by default with console-formatted output. Use `--quiet` to suppress.

## Output

Results are stored in the `runs` table:

```sql
SELECT * FROM runs;
```

| Column | Description |
|--------|-------------|
| `model_type` | `MF` or `FM` |
| `best_type` | Best model variant (e.g. `BPR`, `ALS`, `FM`) |
| `params` | JSON hyperparameters |
| `score` | JSON scores (NDCG/Precision/Recall for MF, AUC/Precision/Recall for FM) |
| `trials` | Number of trials run |
| `duration_seconds` | Wall time for optimization |
| `created_at` | Timestamp |

## Examples

Quick test with 2 trials:

```bash
./gorse-optimize --config config/config.toml --trials 2
```

Full run with custom output path, quiet mode:

```bash
./gorse-optimize --config config/config.toml --trials 50 --patience 15 --output /data/optimize-2026-03-31.sqlite3 --quiet
```

Inspect results:

```bash
sqlite3 optimize-results.sqlite3 "SELECT model_type, best_type, score, duration_seconds FROM runs ORDER BY created_at DESC"
```
