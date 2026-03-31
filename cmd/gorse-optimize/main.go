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

func main() {
	rootCmd := &cobra.Command{
		Use:   "gorse-optimize",
		Short: "Run hyperparameter optimization for Gorse models",
		Run:   run,
	}

	flags := rootCmd.Flags()
	flags.String("config", "", "path to gorse TOML config (required)")
	flags.Int("trials", 20, "number of goptuna trials per model")
	flags.Int("jobs", runtime.NumCPU(), "parallel workers for model fitting")
	flags.Int("patience", 10, "early stopping patience")
	flags.String("output", "./optimize-results.sqlite3", "path for output SQLite database")
	flags.Bool("quiet", false, "suppress log output")
	lo.Must0(rootCmd.MarkFlagRequired("config"))

	log.AddFlags(flags)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func run(cmd *cobra.Command, args []string) {
	quiet, _ := cmd.Flags().GetBool("quiet")
	if quiet {
		log.CloseLogger()
	} else {
		// Use console encoder for human-readable output
		log.SetLogger(cmd.Flags(), true)
	}

	configPath, _ := cmd.Flags().GetString("config")
	trials, _ := cmd.Flags().GetInt("trials")
	jobs, _ := cmd.Flags().GetInt("jobs")
	patience, _ := cmd.Flags().GetInt("patience")
	outputPath, _ := cmd.Flags().GetString("output")

	// Load configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		log.Logger().Fatal("failed to load config", zap.Error(err))
	}

	// Load dataset
	log.Logger().Info("loading data from database...")
	m := master.NewMaster(cfg, os.TempDir(), false, configPath)
	m.DataClient, err = data.Open(m.Config.Database.DataStore, m.Config.Database.DataTablePrefix,
		storage.WithIsolationLevel(m.Config.Database.MySQL.IsolationLevel))
	if err != nil {
		log.Logger().Fatal("failed to open data client", zap.Error(err))
	}
	evaluator := master.NewOnlineEvaluator(
		m.Config.Recommend.DataSource.PositiveFeedbackTypes,
		m.Config.Recommend.DataSource.ReadFeedbackTypes)
	ctrDataset, cfDataset, err := m.LoadDataFromDatabase(context.Background(), m.DataClient,
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

	// Open output SQLite
	db, err := openOutputDB(outputPath)
	if err != nil {
		log.Logger().Fatal("failed to open output database", zap.Error(err))
	}
	defer db.Close()

	// Optimize collaborative filtering
	log.Logger().Info("optimizing collaborative filtering model",
		zap.Int("trials", trials), zap.Int("jobs", jobs), zap.Int("patience", patience))

	cfTrainSet, cfTestSet := cfDataset.SplitCF(0, 0)
	cfResult, cfDuration, err := optimizeCF(cfTrainSet, cfTestSet, trials, jobs, patience)
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

	// Optimize click-through rate prediction
	log.Logger().Info("optimizing click-through rate model",
		zap.Int("trials", trials), zap.Int("jobs", jobs), zap.Int("patience", patience))

	ctrTrainSet, ctrTestSet := ctrDataset.Split(0.2, 0)
	ctrResult, ctrDuration, err := optimizeCTR(ctrTrainSet, ctrTestSet, trials, jobs, patience)
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

	// Print results table
	fmt.Println()
	printResults(cfResult, cfDuration, ctrResult, ctrDuration)
	fmt.Printf("\nResults saved to %s\n", outputPath)
}

func optimizeCF(trainSet, testSet dataset.CFSplit, trials, jobs, patience int) (meta.Model[cf.Score], time.Duration, error) {
	if trainSet.CountUsers() == 0 || trainSet.CountItems() == 0 || trainSet.CountFeedback() == 0 {
		return meta.Model[cf.Score]{}, 0, fmt.Errorf("insufficient data: %d users, %d items, %d feedback",
			trainSet.CountUsers(), trainSet.CountItems(), trainSet.CountFeedback())
	}

	search := cf.NewModelSearch(map[string]cf.ModelCreator{
		"BPR": func() cf.MatrixFactorization { return cf.NewBPR(nil) },
		"ALS": func() cf.MatrixFactorization { return cf.NewALS(nil) },
	}, trainSet, testSet,
		cf.NewFitConfig().SetJobs(jobs).SetPatience(patience))

	trialNum := 0
	objective := func(trial goptuna.Trial) (float64, error) {
		trialNum++
		log.Logger().Info(fmt.Sprintf("CF trial %d/%d starting", trialNum, trials))
		start := time.Now()
		score, err := search.Objective(trial)
		if err != nil {
			log.Logger().Error(fmt.Sprintf("CF trial %d/%d failed", trialNum, trials), zap.Error(err))
			return score, err
		}
		result := search.Result()
		log.Logger().Info(fmt.Sprintf("CF trial %d/%d completed", trialNum, trials),
			zap.Float64("ndcg", score),
			zap.String("best_so_far", fmt.Sprintf("%s (NDCG=%.4f)", result.Type, result.Score.NDCG)),
			zap.String("duration", formatDuration(time.Since(start))))
		return score, nil
	}

	start := time.Now()
	study, err := goptuna.CreateStudy("optimizeCollaborativeFiltering",
		goptuna.StudyOptionDirection(goptuna.StudyDirectionMaximize),
		goptuna.StudyOptionSampler(tpe.NewSampler()),
		goptuna.StudyOptionLogger(newQuietOptunaLogger()))
	if err != nil {
		return meta.Model[cf.Score]{}, 0, err
	}
	if err = study.Optimize(objective, trials); err != nil {
		return meta.Model[cf.Score]{}, 0, err
	}
	return search.Result(), time.Since(start), nil
}

func optimizeCTR(trainSet, testSet *ctr.Dataset, trials, jobs, patience int) (meta.Model[ctr.Score], time.Duration, error) {
	if trainSet.CountUsers() == 0 || trainSet.CountItems() == 0 || trainSet.Count() == 0 {
		return meta.Model[ctr.Score]{}, 0, fmt.Errorf("insufficient data: %d users, %d items, %d interactions",
			trainSet.CountUsers(), trainSet.CountItems(), trainSet.Count())
	}

	search := ctr.NewModelSearch(map[string]ctr.ModelCreator{
		"FM": func() ctr.FactorizationMachines { return ctr.NewAFM(nil) },
	}, trainSet, testSet,
		ctr.NewFitConfig().SetJobs(jobs).SetPatience(patience))

	trialNum := 0
	objective := func(trial goptuna.Trial) (float64, error) {
		trialNum++
		log.Logger().Info(fmt.Sprintf("CTR trial %d/%d starting", trialNum, trials))
		start := time.Now()
		score, err := search.Objective(trial)
		if err != nil {
			log.Logger().Error(fmt.Sprintf("CTR trial %d/%d failed", trialNum, trials), zap.Error(err))
			return score, err
		}
		result := search.Result()
		log.Logger().Info(fmt.Sprintf("CTR trial %d/%d completed", trialNum, trials),
			zap.Float64("auc", score),
			zap.String("best_so_far", fmt.Sprintf("%s (AUC=%.4f)", result.Type, result.Score.AUC)),
			zap.String("duration", formatDuration(time.Since(start))))
		return score, nil
	}

	start := time.Now()
	study, err := goptuna.CreateStudy("optimizeClickThroughRatePrediction",
		goptuna.StudyOptionDirection(goptuna.StudyDirectionMaximize),
		goptuna.StudyOptionSampler(tpe.NewSampler()),
		goptuna.StudyOptionLogger(newQuietOptunaLogger()))
	if err != nil {
		return meta.Model[ctr.Score]{}, 0, err
	}
	if err = study.Optimize(objective, trials); err != nil {
		return meta.Model[ctr.Score]{}, 0, err
	}
	return search.Result(), time.Since(start), nil
}

// quietOptunaLogger suppresses goptuna's own trial logging since we handle it ourselves.
type quietOptunaLogger struct{}

func newQuietOptunaLogger() goptuna.Logger { return &quietOptunaLogger{} }

func (q *quietOptunaLogger) Debug(string, ...interface{}) {}
func (q *quietOptunaLogger) Info(string, ...interface{})  {}
func (q *quietOptunaLogger) Warn(string, ...interface{})  {}
func (q *quietOptunaLogger) Error(string, ...interface{}) {}

func openOutputDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path)
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

func printResults(cfResult meta.Model[cf.Score], cfDuration time.Duration, ctrResult meta.Model[ctr.Score], ctrDuration time.Duration) {
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

