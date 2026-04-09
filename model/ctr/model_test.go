// Copyright 2020 gorse Project Authors
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
package ctr

import (
	"bytes"
	"context"
	"math"
	"runtime"
	"testing"

	"github.com/gorse-io/gorse/common/nn"
	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/model"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

const classificationDelta = 0.01

func newFitConfigWithTestTracker() *FitConfig {
	cfg := NewFitConfig().SetVerbose(1).SetJobs(runtime.NumCPU())
	return cfg
}

func TestFactorizationMachines_Classification_Frappe(t *testing.T) {
	// python .\model.py frappe -dim 8 -iter 10 -learn_rate 0.01 -regular 0.0001
	train, test, err := LoadDataFromBuiltIn("frappe")
	assert.NoError(t, err)
	m := NewAFM(model.Params{
		model.NFactors:  8,
		model.NEpochs:   10,
		model.Lr:        0.01,
		model.Reg:       0.0001,
		model.BatchSize: 1024,
	})
	fitConfig := newFitConfigWithTestTracker()
	score := m.Fit(context.Background(), train, test, fitConfig)
	assert.InDelta(t, 0.919, score.Accuracy, classificationDelta)
}

func TestFactorizationMachines_Classification_MovieLens(t *testing.T) {
	t.Skip("Skip time-consuming test")
	// python .\model.py ml-tag -dim 8 -iter 10 -learn_rate 0.01 -regular 0.0001
	train, test, err := LoadDataFromBuiltIn("ml-tag")
	assert.NoError(t, err)
	m := NewAFM(model.Params{
		model.InitStdDev: 0.01,
		model.NFactors:   8,
		model.NEpochs:    10,
		model.Lr:         0.001,
		model.Reg:        0.0001,
		model.BatchSize:  1024,
	})
	fitConfig := newFitConfigWithTestTracker()
	score := m.Fit(context.Background(), train, test, fitConfig)
	assert.InDelta(t, 0.815, score.Accuracy, classificationDelta)
}

func TestFactorizationMachines_Classification_Criteo(t *testing.T) {
	// python .\model.py criteo -dim 8 -iter 10 -learn_rate 0.01 -regular 0.0001
	train, test, err := LoadDataFromBuiltIn("criteo")
	assert.NoError(t, err)
	m := NewAFM(model.Params{
		model.NFactors:  8,
		model.NEpochs:   10,
		model.Lr:        0.01,
		model.Reg:       0.0001,
		model.BatchSize: 1024,
	})
	fitConfig := newFitConfigWithTestTracker()
	score := m.Fit(context.Background(), train, test, fitConfig)
	assert.InDelta(t, 0.77, score.Accuracy, 0.025)

	// test prediction
	assert.Equal(t,
		m.BatchInternalPredict(
			[]lo.Tuple2[[]int32, []float32]{{A: []int32{1, 2, 3, 4, 5, 6}, B: []float32{1, 1, 0.3, 0.4, 0.5, 0.6}}},
			make([][][]float32, 2), fitConfig.Jobs),
		m.BatchPredict([]lo.Tuple4[string, string, []Label, []Label]{{
			A: "1",
			B: "2",
			C: []Label{
				{Name: "3", Value: 0.3},
				{Name: "4", Value: 0.4},
			},
			D: []Label{
				{Name: "5", Value: 0.5},
				{Name: "6", Value: 0.6},
			}}}, make([][]Embedding, 2), fitConfig.Jobs))

	// test marshal and unmarshal
	buf := bytes.NewBuffer(nil)
	err = MarshalModel(buf, m)
	assert.NoError(t, err)
	tmp, err := UnmarshalModel(buf)
	assert.NoError(t, err)
	scoreClone := EvaluateClassification(tmp, test, fitConfig.Jobs)
	assert.InDelta(t, 0.77, scoreClone.Accuracy, 0.02)

	// test clear
	assert.False(t, m.Invalid())
	m.Clear()
	assert.True(t, m.Invalid())
}

func newSynthesisDataset() *Dataset {
	builder := dataset.NewUnifiedMapIndexBuilder()
	builder.AddUser("u0")
	builder.AddUser("u1")
	builder.AddUserLabel("ul0")
	builder.AddUserLabel("ul1")
	builder.AddUserLabel("ul2")
	builder.AddItem("i0")
	builder.AddItem("i1")
	builder.AddItemLabel("il0")
	builder.AddItemLabel("il1")
	builder.AddItemLabel("il2")

	dataSet := NewMapIndexDataset()
	dataSet.Index = builder.Build()
	dataSet.UserLabels = [][]lo.Tuple2[int32, float32]{
		{{A: 0, B: 1.0}, {A: 1, B: 0.5}, {A: 2, B: -1.0}},
		{{A: 0, B: -1.0}, {A: 1, B: -0.5}, {A: 2, B: 1.0}},
	}
	dataSet.ItemLabels = [][]lo.Tuple2[int32, float32]{
		{{A: 0, B: 1.0}, {A: 1, B: 0.5}, {A: 2, B: -1.0}},
		{{A: 0, B: -1.0}, {A: 1, B: -0.5}, {A: 2, B: 1.0}},
	}
	dataSet.ItemEmbeddingIndex = dataset.NewMapIndex()
	dataSet.ItemEmbeddingIndex.Add("e1")
	dataSet.ItemEmbeddingIndex.Add("e2")
	dataSet.ItemEmbeddingDimension = []int{3, 4}
	dataSet.ItemEmbeddings = [][][]float32{
		{{0.8, 0.8, 0.8}, {0.1, 0.1, 0.1, 0.1}},
		{{-0.8, -0.8, -0.8}, {-0.1, -0.1, -0.1, -0.1}},
	}

	dataSet.Users = []int32{0, 0, 1, 1}
	dataSet.Items = []int32{0, 1, 0, 1}
	dataSet.Target = []float32{1, -1, -1, 1}
	dataSet.PositiveCount = 2
	dataSet.NegativeCount = 2
	return dataSet
}

func TestFactorizationMachines_Classification_Synthesis(t *testing.T) {
	dataSet := newSynthesisDataset()
	fitConfig := newFitConfigWithTestTracker()
	m := NewAFM(nil)
	score := m.Fit(context.Background(), dataSet, dataSet, fitConfig)
	assert.GreaterOrEqual(t, score.Accuracy, float32(0.5))

	buf := bytes.NewBuffer(nil)
	err := MarshalModel(buf, m)
	assert.NoError(t, err)
	clone, err := UnmarshalModel(buf)
	assert.NoError(t, err)
	cloneScore := EvaluateClassification(clone, dataSet, fitConfig.Jobs)
	assert.InDelta(t, score.Accuracy, cloneScore.Accuracy, 0.05)

	indicesPos, valuesPos, embeddingsPos, _ := dataSet.Get(0)
	indicesNeg, valuesNeg, embeddingsNeg, _ := dataSet.Get(1)
	assert.Equal(t,
		m.BatchInternalPredict(
			[]lo.Tuple2[[]int32, []float32]{
				{A: indicesPos, B: valuesPos},
				{A: indicesNeg, B: valuesNeg},
			},
			[][][]float32{embeddingsPos, embeddingsNeg},
			fitConfig.Jobs,
		),
		m.BatchPredict(
			[]lo.Tuple4[string, string, []Label, []Label]{
				{
					A: "u0",
					B: "i0",
					C: []Label{{Name: "ul0", Value: 1.0}, {Name: "ul1", Value: 0.5}, {Name: "ul2", Value: -1.0}},
					D: []Label{{Name: "il0", Value: 1.0}, {Name: "il1", Value: 0.5}, {Name: "il2", Value: -1.0}},
				},
				{
					A: "u0",
					B: "i1",
					C: []Label{{Name: "ul0", Value: 1.0}, {Name: "ul1", Value: 0.5}, {Name: "ul2", Value: -1.0}},
					D: []Label{{Name: "il0", Value: -1.0}, {Name: "il1", Value: -0.5}, {Name: "il2", Value: 1.0}},
				},
			},
			[][]Embedding{
				{{Name: "e1", Value: embeddingsPos[0]}, {Name: "e2", Value: embeddingsPos[1]}},
				{{Name: "e1", Value: embeddingsNeg[0]}, {Name: "e2", Value: embeddingsNeg[1]}},
			},
			fitConfig.Jobs,
		))

	assert.Len(t, m.BatchPredict(
		[]lo.Tuple4[string, string, []Label, []Label]{
			{A: "u0", B: "i0"},
			{A: "u0", B: "i1"},
			{
				A: "u0",
				B: "i0",
				C: []Label{{Name: "ul_unknown", Value: 1}},
				D: []Label{{Name: "il_unknown", Value: 1}},
			},
			{
				A: "u0",
				B: "i1",
				C: []Label{{Name: "ul_unknown", Value: 1}},
				D: []Label{{Name: "il_unknown", Value: 1}},
			},
		},
		[][]Embedding{
			{},
			{},
			{{Name: "unknown_embedding", Value: make([]float32, 3)}},
			{{Name: "unknown_embedding", Value: make([]float32, 3)}},
		},
		fitConfig.Jobs,
	), 4)
}

// newModifiedSynthesisDataset creates a dataset with one user and one item
// added/removed compared to newSynthesisDataset, to test index remapping.
func newModifiedSynthesisDataset() *Dataset {
	builder := dataset.NewUnifiedMapIndexBuilder()
	builder.AddUser("u0")
	builder.AddUser("u2") // u1 removed, u2 added
	builder.AddUserLabel("ul0")
	builder.AddUserLabel("ul1")
	builder.AddUserLabel("ul2")
	builder.AddItem("i0")
	builder.AddItem("i2") // i1 removed, i2 added
	builder.AddItemLabel("il0")
	builder.AddItemLabel("il1")
	builder.AddItemLabel("il2")

	dataSet := NewMapIndexDataset()
	dataSet.Index = builder.Build()
	dataSet.UserLabels = [][]lo.Tuple2[int32, float32]{
		{{A: 0, B: 1.0}, {A: 1, B: 0.5}, {A: 2, B: -1.0}},
		{{A: 0, B: -1.0}, {A: 1, B: -0.5}, {A: 2, B: 1.0}},
	}
	dataSet.ItemLabels = [][]lo.Tuple2[int32, float32]{
		{{A: 0, B: 1.0}, {A: 1, B: 0.5}, {A: 2, B: -1.0}},
		{{A: 0, B: -1.0}, {A: 1, B: -0.5}, {A: 2, B: 1.0}},
	}
	dataSet.ItemEmbeddingIndex = dataset.NewMapIndex()
	dataSet.ItemEmbeddingIndex.Add("e1")
	dataSet.ItemEmbeddingIndex.Add("e2")
	dataSet.ItemEmbeddingDimension = []int{3, 4}
	dataSet.ItemEmbeddings = [][][]float32{
		{{0.8, 0.8, 0.8}, {0.1, 0.1, 0.1, 0.1}},
		{{-0.8, -0.8, -0.8}, {-0.1, -0.1, -0.1, -0.1}},
	}

	dataSet.Users = []int32{0, 0, 1, 1}
	dataSet.Items = []int32{0, 1, 0, 1}
	dataSet.Target = []float32{1, -1, -1, 1}
	dataSet.PositiveCount = 2
	dataSet.NegativeCount = 2
	return dataSet
}

func TestAFM_WarmStart_FeatureRemapping(t *testing.T) {
	// Train on the original dataset
	original := newSynthesisDataset()
	fitConfig := newFitConfigWithTestTracker()
	m := NewAFM(model.Params{model.NEpochs: 5})
	m.Fit(context.Background(), original, original, fitConfig)

	// Marshal/unmarshal to simulate blob storage round-trip
	buf := bytes.NewBuffer(nil)
	err := MarshalModel(buf, m)
	assert.NoError(t, err)
	loaded, err := UnmarshalModel(buf)
	assert.NoError(t, err)
	oldModel := loaded.(*AFM)

	// Warm-start on modified dataset (u1→u2, i1→i2)
	modified := newModifiedSynthesisDataset()
	warm := NewAFM(model.Params{model.NEpochs: 2})
	warm.SetWarmModel(oldModel)
	warm.Fit(context.Background(), modified, modified, fitConfig)

	// Verify: shared features (u0, i0, labels) should have been copied.
	// The bias should match the old model's trained bias.
	assert.InDelta(t, oldModel.B.Data()[0], warm.B.Data()[0], 0.5,
		"bias should be close to old model after warm-start + 2 epochs")

	// Verify: new features (u2, i2) should exist and have valid weights.
	newW := warm.W.Parameters()[0]
	u2Idx := modified.Index.EncodeUser("u2")
	assert.NotEqual(t, dataset.NotId, u2Idx)
	assert.False(t, math.IsNaN(float64(newW.Data()[u2Idx])),
		"new user u2 should have a valid weight")
}

func TestAFM_WarmStart_NFactorsMismatch(t *testing.T) {
	// Train with nFactors=8
	original := newSynthesisDataset()
	fitConfig := newFitConfigWithTestTracker()
	m := NewAFM(model.Params{model.NFactors: 8, model.NEpochs: 3})
	m.Fit(context.Background(), original, original, fitConfig)

	// Marshal/unmarshal
	buf := bytes.NewBuffer(nil)
	err := MarshalModel(buf, m)
	assert.NoError(t, err)
	loaded, err := UnmarshalModel(buf)
	assert.NoError(t, err)
	oldModel := loaded.(*AFM)
	assert.Equal(t, 8, oldModel.nFactors)

	// Warm-start with nFactors=16 (different)
	warm := NewAFM(model.Params{model.NFactors: 16, model.NEpochs: 2})
	warm.SetWarmModel(oldModel)
	warm.Fit(context.Background(), original, original, fitConfig)

	// A/E layers should NOT have been copied (nFactors differs).
	// Verify the model still works (no panic, valid output).
	score := EvaluateClassification(warm, original, fitConfig.Jobs)
	assert.False(t, math.IsNaN(float64(score.AUC)), "model should produce valid AUC")

	// V embeddings: only min(8,16)=8 factors should have been copied.
	// The remaining 8 factors should be random-initialized (non-zero for at least some).
	newV := warm.V.Parameters()[0]
	vShape := newV.Shape()
	assert.Equal(t, 16, vShape[1], "new V should have 16 factors")

	// Check that a shared feature's first 8 factors are from old model.
	// Both models use the same dataset, so the index for u0 is the same.
	u0Idx := original.Index.EncodeUser("u0")
	oldV := oldModel.V.Parameters()[0]
	for f := 0; f < 8; f++ {
		oldVal := oldV.Data()[int(u0Idx)*8+f]
		newVal := newV.Data()[int(u0Idx)*16+f]
		assert.InDelta(t, oldVal, newVal, 0.5,
			"first 8 factors of u0 should be close to old model after 2 epochs")
	}

	// A layers should have fresh random weights, not old model's
	oldAParams := oldModel.A[0].Parameters()
	newAParams := warm.A[0].Parameters()
	if len(oldAParams) > 0 && len(newAParams) > 0 {
		assert.NotEqual(t, len(oldAParams[0].Data()), len(newAParams[0].Data()),
			"A layer tensor sizes should differ when nFactors changes")
	}
}

func TestAFM_WarmStart_CopyLayerParams(t *testing.T) {
	// Happy path: matching layers
	dst := nn.NewEmbedding(5, 3)
	src := nn.NewEmbedding(5, 3)
	copyLayerParams(dst, src)
	assert.Equal(t, src.Parameters()[0].Data(), dst.Parameters()[0].Data())

	// Mismatch: Attention has 2+ parameters, Embedding has 1.
	// copyLayerParams should skip without panicking.
	embedding := nn.NewEmbedding(5, 3)
	attention := nn.NewAttention(3, 4)
	dataBefore := make([]float32, len(embedding.Parameters()[0].Data()))
	copy(dataBefore, embedding.Parameters()[0].Data())
	copyLayerParams(embedding, attention) // should warn and skip
	assert.Equal(t, dataBefore, embedding.Parameters()[0].Data(),
		"destination should be unchanged when parameter counts differ")
}
