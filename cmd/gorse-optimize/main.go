// Copyright 2025 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/tpe"
	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/master"
	"github.com/gorse-io/gorse/model"
	"github.com/gorse-io/gorse/model/cf"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/meta"
	"github.com/olekukonko/tablewriter"
	"github.com/samber/lo"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	_ "modernc.org/sqlite"
)

const metaTimeout = 10 * time.Second

func main() {
	rootCmd := &cobra.Command{
		Use:   "gorse-optimize",
		Short: "Hyperparameter optimization for Gorse models",
	}

	pflags := rootCmd.PersistentFlags()
	pflags.String("output", "./optimize-results.sqlite3", "path for results SQLite database")
	pflags.String("cache-path", config.MkDir("master"), "path of cache folder containing meta.sqlite3")

	rootCmd.AddCommand(newRunCmd())
	rootCmd.AddCommand(newListCmd())
	rootCmd.AddCommand(newApplyCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// --- run subcommand ---

func newRunCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run hyperparameter optimization",
		Run:   runOptimize,
	}
	flags := cmd.Flags()
	flags.String("config", "", "path to gorse TOML config (required)")
	flags.Int("trials", 10, "number of goptuna trials per model")
	flags.Int("jobs", runtime.NumCPU(), "parallel workers for model fitting")
	flags.Int("patience", 10, "early stopping patience")
	flags.Float64("split-ratio", 0.2, "fraction of CTR data used for testing (0.0-1.0)")
	flags.Bool("quiet", false, "suppress log output")
	lo.Must0(cmd.MarkFlagRequired("config"))
	log.AddFlags(flags)
	return cmd
}

func runOptimize(cmd *cobra.Command, args []string) {
	quiet, _ := cmd.Flags().GetBool("quiet")
	if quiet {
		log.CloseLogger()
	} else {
		log.SetLogger(cmd.Flags(), true)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	configPath, _ := cmd.Flags().GetString("config")
	trials, _ := cmd.Flags().GetInt("trials")
	jobs, _ := cmd.Flags().GetInt("jobs")
	patience, _ := cmd.Flags().GetInt("patience")
	outputPath, _ := cmd.Flags().GetString("output")
	splitRatio, _ := cmd.Flags().GetFloat64("split-ratio")

	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Logger().Fatal("failed to load config", zap.Error(err))
	}

	log.Logger().Info("loading data from database")
	m := master.NewMaster(cfg, os.TempDir(), false, configPath)
	m.DataClient, err = data.Open(m.Config.Database.DataStore, m.Config.Database.DataTablePrefix,
		storage.WithIsolationLevel(m.Config.Database.MySQL.IsolationLevel))
	if err != nil {
		log.Logger().Fatal("failed to open data client", zap.Error(err))
	}
	evaluator := master.NewOnlineEvaluator(
		m.Config.Recommend.DataSource.PositiveFeedbackTypes,
		m.Config.Recommend.DataSource.ReadFeedbackTypes)
	ctrDataset, cfDataset, err := m.LoadDataFromDatabase(ctx, m.DataClient,
		m.Config.Recommend.DataSource.PositiveFeedbackTypes,
		m.Config.Recommend.DataSource.ReadFeedbackTypes,
		m.Config.Recommend.DataSource.ItemTTL,
		m.Config.Recommend.DataSource.PositiveFeedbackTTL,
		evaluator,
		nil)
	if err != nil {
		log.Logger().Fatal("failed to load dataset", zap.Error(err))
	}

	log.Logger().Info("dataset loaded",
		zap.Int("users", cfDataset.CountUsers()),
		zap.Int("items", cfDataset.CountItems()),
		zap.Int("feedback", cfDataset.CountFeedback()))

	db, err := openOutputDB(outputPath)
	if err != nil {
		log.Logger().Fatal("failed to open output database", zap.Error(err))
	}
	defer db.Close()

	log.Logger().Info("optimizing collaborative filtering model",
		zap.Int("trials", trials), zap.Int("jobs", jobs), zap.Int("patience", patience))

	cfTrainSet, cfTestSet := cfDataset.SplitCF(0, 0)
	cfResult, cfDuration, err := optimizeCF(ctx, cfTrainSet, cfTestSet, trials, jobs, patience)
	if err != nil {
		log.Logger().Fatal("failed to optimize CF model", zap.Error(err))
	}
	log.Logger().Info("collaborative filtering optimization completed",
		zap.String("best_type", cfResult.Type),
		zap.String("params", formatParams(cfResult.Params)),
		zap.String("score", formatCFScore(cfResult.Score)),
		zap.String("duration", formatDuration(cfDuration)))

	if err := saveRun(db, "MF", cfResult.Type, cfResult.Params, cfResult.Score, trials, cfDuration); err != nil {
		log.Logger().Fatal("failed to save CF result", zap.Error(err))
	}

	log.Logger().Info("optimizing click-through rate model",
		zap.Int("trials", trials), zap.Int("jobs", jobs), zap.Int("patience", patience))

	ctrTrainSet, ctrTestSet := ctrDataset.Split(float32(splitRatio), 0)
	ctrResult, ctrDuration, err := optimizeCTR(ctx, ctrTrainSet, ctrTestSet, trials, jobs, patience)
	if err != nil {
		log.Logger().Fatal("failed to optimize CTR model", zap.Error(err))
	}
	log.Logger().Info("click-through rate optimization completed",
		zap.String("best_type", ctrResult.Type),
		zap.String("params", formatParams(ctrResult.Params)),
		zap.String("score", formatCTRScore(ctrResult.Score)),
		zap.String("duration", formatDuration(ctrDuration)))

	if err := saveRun(db, "FM", ctrResult.Type, ctrResult.Params, ctrResult.Score, trials, ctrDuration); err != nil {
		log.Logger().Fatal("failed to save CTR result", zap.Error(err))
	}

	fmt.Println()
	printRunResults(cfResult, cfDuration, ctrResult, ctrDuration)
	fmt.Printf("\nResults saved to %s\n", outputPath)
}

// --- list subcommand ---

func newListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List optimization runs and current meta store values",
		Run:   runList,
	}
	cmd.Flags().Int("limit", 20, "max number of runs to show")
	return cmd
}

func runList(cmd *cobra.Command, args []string) {
	outputPath, _ := cmd.Flags().GetString("output")
	cacheDir, _ := cmd.Flags().GetString("cache-path")
	limit, _ := cmd.Flags().GetInt("limit")

	printCurrentMeta(cacheDir)

	db, err := openOutputDBReadOnly(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening results database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	rows, err := db.Query(`SELECT id, model_type, best_type, params, score, trials, duration_seconds, created_at
		FROM runs ORDER BY created_at DESC LIMIT ?`, limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error querying runs: %v\n", err)
		os.Exit(1)
	}
	defer rows.Close()

	fmt.Println("Optimization runs:")
	table := tablewriter.NewWriter(os.Stdout)
	table.Header([]string{"ID", "Model", "Best Type", "Score", "Params", "Duration", "Date"})
	hasRows := false
	for rows.Next() {
		hasRows = true
		var id, trials int
		var modelType, bestType, paramsJSON, scoreJSON, createdAt string
		var durationSec float64
		if err := rows.Scan(&id, &modelType, &bestType, &paramsJSON, &scoreJSON, &trials, &durationSec, &createdAt); err != nil {
			fmt.Fprintf(os.Stderr, "Error scanning row: %v\n", err)
			os.Exit(1)
		}
		score := formatStoredScore(modelType, scoreJSON)
		params := formatStoredParams(paramsJSON)
		duration := formatDuration(time.Duration(durationSec * float64(time.Second)))
		// truncate timestamp to minute
		for _, layout := range []string{"2006-01-02 15:04:05", time.RFC3339, time.RFC3339Nano} {
			if t, err := time.Parse(layout, createdAt); err == nil {
				createdAt = t.Local().Format("2006-01-02 15:04")
				break
			}
		}
		lo.Must0(table.Append([]string{
			fmt.Sprintf("%d", id), modelType, bestType, score, params, duration, createdAt,
		}))
	}
	if err := rows.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error iterating runs: %v\n", err)
		os.Exit(1)
	}
	if !hasRows {
		fmt.Println("  (no runs found)")
	} else {
		lo.Must0(table.Render())
	}
}

func printCurrentMeta(cacheDir string) {
	metaPath := filepath.Join(cacheDir, "meta.sqlite3")
	metaStore, err := meta.Open(fmt.Sprintf("sqlite://%s", metaPath), metaTimeout)
	if err != nil {
		fmt.Printf("Could not open meta store at %s (use --cache-dir to specify)\n\n", metaPath)
		return
	}
	defer metaStore.Close()

	fmt.Println("Current meta store values:")
	table := tablewriter.NewWriter(os.Stdout)
	table.Header([]string{"Model", "Type", "Score", "Params", "Updated"})

	var rows [][]string
	if cfModel, ok := loadMetaModel[cf.Score](metaStore, meta.COLLABORATIVE_FILTERING_MODEL); ok {
		rows = append(rows, []string{
			"MF",
			cfModel.Type,
			formatCFScore(cfModel.Score),
			formatParams(cfModel.Params),
			formatMetaTimestamp(cfModel.ID),
		})
	}
	if ctrModel, ok := loadMetaModel[ctr.Score](metaStore, meta.CLICK_THROUGH_RATE_MODEL); ok {
		rows = append(rows, []string{
			"FM",
			ctrModel.Type,
			formatCTRScore(ctrModel.Score),
			formatParams(ctrModel.Params),
			formatMetaTimestamp(ctrModel.ID),
		})
	}

	if len(rows) == 0 {
		fmt.Println("  (no models stored)")
	} else {
		lo.Must0(table.Bulk(rows))
		lo.Must0(table.Render())
	}
	fmt.Println()
}

func loadMetaModel[T any](metaStore meta.Database, key string) (meta.Model[T], bool) {
	var m meta.Model[T]
	str, err := metaStore.Get(key)
	if err != nil || str == nil {
		return m, false
	}
	if err := m.FromJSON(*str); err != nil {
		return m, false
	}
	return m, true
}

func formatMetaTimestamp(unixSec int64) string {
	if unixSec == 0 {
		return "-"
	}
	return time.Unix(unixSec, 0).Format("2006-01-02 15:04")
}

// --- apply subcommand ---

func newApplyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "apply",
		Short: "Apply optimization results to the gorse meta store",
		Run:   runApply,
	}
	cmd.Flags().Int("run-id", 0, "specific run ID to apply (default: latest of each model type)")
	return cmd
}

func runApply(cmd *cobra.Command, args []string) {
	outputPath, _ := cmd.Flags().GetString("output")
	cacheDir, _ := cmd.Flags().GetString("cache-path")
	runID, _ := cmd.Flags().GetInt("run-id")

	db, err := openOutputDBReadOnly(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening results database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	metaPath := filepath.Join(cacheDir, "meta.sqlite3")
	metaStore, err := meta.Open(fmt.Sprintf("sqlite://%s", metaPath), metaTimeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening meta store at %s: %v\n", metaPath, err)
		os.Exit(1)
	}
	defer metaStore.Close()

	if runID > 0 {
		applyRunByID(db, metaStore, runID)
	} else {
		applyLatest(db, metaStore, "MF", meta.COLLABORATIVE_FILTERING_MODEL)
		applyLatest(db, metaStore, "FM", meta.CLICK_THROUGH_RATE_MODEL)
	}
}

func applyRunByID(db *sql.DB, metaStore meta.Database, runID int) {
	var modelType, bestType, paramsJSON, scoreJSON string
	err := db.QueryRow(`SELECT model_type, best_type, params, score FROM runs WHERE id = ?`, runID).
		Scan(&modelType, &bestType, &paramsJSON, &scoreJSON)
	if err == sql.ErrNoRows {
		fmt.Fprintf(os.Stderr, "Run ID %d not found\n", runID)
		os.Exit(1)
	} else if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading run: %v\n", err)
		os.Exit(1)
	}

	var metaKey string
	switch modelType {
	case "MF":
		metaKey = meta.COLLABORATIVE_FILTERING_MODEL
	case "FM":
		metaKey = meta.CLICK_THROUGH_RATE_MODEL
	default:
		fmt.Fprintf(os.Stderr, "Unknown model type: %s\n", modelType)
		os.Exit(1)
	}

	applyToMeta(metaStore, metaKey, modelType, bestType, paramsJSON, scoreJSON)
}

func applyLatest(db *sql.DB, metaStore meta.Database, modelType, metaKey string) {
	var bestType, paramsJSON, scoreJSON string
	err := db.QueryRow(`SELECT best_type, params, score FROM runs WHERE model_type = ? ORDER BY created_at DESC LIMIT 1`, modelType).
		Scan(&bestType, &paramsJSON, &scoreJSON)
	if err == sql.ErrNoRows {
		fmt.Printf("No %s runs found, skipping\n", modelType)
		return
	} else if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading latest %s run: %v\n", modelType, err)
		os.Exit(1)
	}

	applyToMeta(metaStore, metaKey, modelType, bestType, paramsJSON, scoreJSON)
}

func applyToMeta(metaStore meta.Database, metaKey, modelType, bestType, paramsJSON, scoreJSON string) {
	// Build meta JSON manually because we're type-erased here (params/score are raw JSON
	// from the results DB) and can't instantiate a typed meta.Model[T] without knowing T.
	metaJSON, err := json.Marshal(map[string]interface{}{
		"ID":     time.Now().Unix(),
		"Type":   bestType,
		"Params": json.RawMessage(paramsJSON),
		"Score":  json.RawMessage(scoreJSON),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error building meta JSON for %s: %v\n", modelType, err)
		os.Exit(1)
	}

	if err := metaStore.Put(metaKey, string(metaJSON)); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing %s to meta store: %v\n", modelType, err)
		os.Exit(1)
	}

	params := formatStoredParams(paramsJSON)
	score := formatStoredScore(modelType, scoreJSON)
	fmt.Printf("Applied %s: type=%s, score=%s, params=%s\n", modelType, bestType, score, params)
}

// --- optimization helpers ---

func optimizeCF(ctx context.Context, trainSet, testSet dataset.CFSplit, trials, jobs, patience int) (meta.Model[cf.Score], time.Duration, error) {
	if trainSet.CountUsers() == 0 || trainSet.CountItems() == 0 || trainSet.CountFeedback() == 0 {
		return meta.Model[cf.Score]{}, 0, fmt.Errorf("insufficient data: %d users, %d items, %d feedback",
			trainSet.CountUsers(), trainSet.CountItems(), trainSet.CountFeedback())
	}

	search := cf.NewModelSearch(map[string]cf.ModelCreator{
		"BPR": func() cf.MatrixFactorization { return cf.NewBPR(nil) },
		"ALS": func() cf.MatrixFactorization { return cf.NewALS(nil) },
	}, trainSet, testSet,
		cf.NewFitConfig().SetJobs(jobs).SetPatience(patience)).
		WithContext(ctx)

	objective := func(trial goptuna.Trial) (float64, error) {
		num, _ := trial.Number()
		log.Logger().Info("CF trial starting", zap.Int("trial", num+1), zap.Int("total", trials))
		start := time.Now()
		score, err := search.Objective(trial)
		if err != nil {
			log.Logger().Error("CF trial failed", zap.Int("trial", num+1), zap.Error(err))
			return score, err
		}
		result := search.Result()
		log.Logger().Info("CF trial completed",
			zap.Int("trial", num+1),
			zap.Float64("ndcg", score),
			zap.String("best_so_far", fmt.Sprintf("%s (NDCG=%.4f)", result.Type, result.Score.NDCG)),
			zap.String("duration", formatDuration(time.Since(start))))
		return score, nil
	}

	start := time.Now()
	if err := runStudy("optimizeCollaborativeFiltering", objective, trials); err != nil {
		return meta.Model[cf.Score]{}, 0, err
	}
	return search.Result(), time.Since(start), nil
}

func optimizeCTR(ctx context.Context, trainSet, testSet *ctr.Dataset, trials, jobs, patience int) (meta.Model[ctr.Score], time.Duration, error) {
	if trainSet.CountUsers() == 0 || trainSet.CountItems() == 0 || trainSet.Count() == 0 {
		return meta.Model[ctr.Score]{}, 0, fmt.Errorf("insufficient data: %d users, %d items, %d interactions",
			trainSet.CountUsers(), trainSet.CountItems(), trainSet.Count())
	}

	search := ctr.NewModelSearch(map[string]ctr.ModelCreator{
		"FM": func() ctr.FactorizationMachines { return ctr.NewAFM(nil) },
	}, trainSet, testSet,
		ctr.NewFitConfig().SetJobs(jobs).SetPatience(patience)).
		WithContext(ctx)

	objective := func(trial goptuna.Trial) (float64, error) {
		num, _ := trial.Number()
		log.Logger().Info("CTR trial starting", zap.Int("trial", num+1), zap.Int("total", trials))
		start := time.Now()
		score, err := search.Objective(trial)
		if err != nil {
			log.Logger().Error("CTR trial failed", zap.Int("trial", num+1), zap.Error(err))
			return score, err
		}
		result := search.Result()
		log.Logger().Info("CTR trial completed",
			zap.Int("trial", num+1),
			zap.Float64("auc", score),
			zap.String("best_so_far", fmt.Sprintf("%s (AUC=%.4f)", result.Type, result.Score.AUC)),
			zap.String("duration", formatDuration(time.Since(start))))
		return score, nil
	}

	start := time.Now()
	if err := runStudy("optimizeClickThroughRatePrediction", objective, trials); err != nil {
		return meta.Model[ctr.Score]{}, 0, err
	}
	return search.Result(), time.Since(start), nil
}

func runStudy(name string, objective goptuna.FuncObjective, trials int) error {
	study, err := goptuna.CreateStudy(name,
		goptuna.StudyOptionDirection(goptuna.StudyDirectionMaximize),
		goptuna.StudyOptionSampler(tpe.NewSampler()),
		goptuna.StudyOptionLogger(log.NewOptunaLogger(log.Logger())))
	if err != nil {
		return err
	}
	return study.Optimize(objective, trials)
}

// --- database helpers ---

func openOutputDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path+"?_pragma=journal_mode(wal)&_pragma=busy_timeout(10000)")
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS runs (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		model_type TEXT,
		best_type TEXT,
		params TEXT,
		score TEXT,
		trials INTEGER,
		duration_seconds REAL,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)`)
	if err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

func openOutputDBReadOnly(path string) (*sql.DB, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("results database not found at %s", path)
	}
	return sql.Open("sqlite", path+"?_pragma=journal_mode(wal)&_pragma=busy_timeout(10000)")
}

func saveRun(db *sql.DB, modelType, bestType string, params interface{}, score interface{}, trials int, duration time.Duration) error {
	paramsJSON, err := json.Marshal(params)
	if err != nil {
		return err
	}
	scoreJSON, err := json.Marshal(score)
	if err != nil {
		return err
	}
	_, err = db.Exec(`INSERT INTO runs (model_type, best_type, params, score, trials, duration_seconds) VALUES (?, ?, ?, ?, ?, ?)`,
		modelType, bestType, string(paramsJSON), string(scoreJSON), trials, duration.Seconds())
	return err
}

// --- formatting helpers ---

func printRunResults(cfResult meta.Model[cf.Score], cfDuration time.Duration, ctrResult meta.Model[ctr.Score], ctrDuration time.Duration) {
	table := tablewriter.NewWriter(os.Stdout)
	table.Header([]string{"Model", "Best Type", "Score", "Params", "Duration"})
	lo.Must0(table.Bulk([][]string{
		{
			"MF",
			cfResult.Type,
			formatCFScore(cfResult.Score),
			formatParams(cfResult.Params),
			formatDuration(cfDuration),
		},
		{
			"FM",
			ctrResult.Type,
			formatCTRScore(ctrResult.Score),
			formatParams(ctrResult.Params),
			formatDuration(ctrDuration),
		},
	}))
	lo.Must0(table.Render())
}

func formatParams(params model.Params) string {
	if len(params) == 0 {
		return "-"
	}
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, string(k))
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		v := params[model.ParamName(k)]
		switch val := v.(type) {
		case float64:
			parts = append(parts, fmt.Sprintf("%s=%.4g", k, val))
		case float32:
			parts = append(parts, fmt.Sprintf("%s=%.4g", k, val))
		default:
			parts = append(parts, fmt.Sprintf("%s=%v", k, v))
		}
	}
	return strings.Join(parts, ", ")
}

func formatStoredParams(paramsJSON string) string {
	var params model.Params
	if err := json.Unmarshal([]byte(paramsJSON), &params); err != nil {
		return paramsJSON
	}
	return formatParams(params)
}

func formatStoredScore(modelType, scoreJSON string) string {
	switch modelType {
	case "MF":
		var s cf.Score
		if err := json.Unmarshal([]byte(scoreJSON), &s); err != nil {
			return scoreJSON
		}
		return formatCFScore(s)
	case "FM":
		var s ctr.Score
		if err := json.Unmarshal([]byte(scoreJSON), &s); err != nil {
			return scoreJSON
		}
		return formatCTRScore(s)
	default:
		return scoreJSON
	}
}

func formatCFScore(s cf.Score) string {
	return fmt.Sprintf("NDCG=%.4f, Precision=%.4f, Recall=%.4f", s.NDCG, s.Precision, s.Recall)
}

func formatCTRScore(s ctr.Score) string {
	return fmt.Sprintf("AUC=%.4f, Precision=%.4f, Recall=%.4f", s.AUC, s.Precision, s.Recall)
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
}
