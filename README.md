



# Clinical Baseline Benchmark

This repository adapts three static single-cell multimodal methods to the PEAG paper's clinical benchmark for metabolomics imputation:

- `MIDAS`  
- `scVAEIT`  
- `StabMap`  

---

## 📌 Benchmark Definition

This benchmark strictly replicates the PEAG paper's **static baseline protocol**:

| Component | Specification |
|-----------|---------------|
| **Sample Unit** | Each clinical visit treated as an independent sample |
| **Training Set** | Paired visits (lab + metabolomics) from training patients only |
| **Test Set** | Held-out patients (never seen during training) |
| **Test Input** | Current visit's laboratory tests only |
| **Prediction Target** | Reconstruct current visit's metabolomics profile |
| **Critical Constraints** | ❌ No autoregressive recurrence❌ No historical state input❌ No cross-visit temporal encoding |

> 💡 This design validates methods as **static multimodal baselines** for clinical imputation tasks.

---

## 📥 Input Format

Two input formats supported:

### Option 1: CSV Directory
```
/path/to/split_dir/
├── train_lab.csv      # Training lab tests (first column = sample ID)
├── train_metab.csv    # Training metabolomics
├── test_lab.csv       # Test lab tests
└── test_metab.csv     # Test metabolomics (ground truth for evaluation)
```
- First column = sample index  
- Remaining columns = feature names  
- Rows = samples, Columns = features  

### Option 2: NPZ Archive
```python
{
  "train_lab": ndarray, 
  "train_metab": ndarray,
  "test_lab": ndarray,
  "test_metab": ndarray
}
```

---

## 📂 Repository Structure

```
clinical_baseline_benchmark/
├── data.py               # Data loading, standardization, export utilities
├── scvaeit_adapter.py    # Gaussian two-block scVAEIT wrapper (lab → metabolomics)
├── midas_adapter.py      # MIDAS wrapper with runtime Gaussian decoder registration
├── metrics.py            # Pearson correlation, MAE, MSE evaluation
scripts/
├── prepare_tabular_benchmark.py  # Export standardized matrices & scaler stats
├── run_scvaeit.py                # Train → predict → inverse transform → evaluate
├── run_midas.py                  # Train → predict → inverse transform → evaluate
├── evaluate_predictions.py       # Generic prediction evaluation
r/
└── run_stabmap_benchmark.R       # StabMap reference/query embedding + KNN imputation
baseline-model/                   # External model implementations (user-provided)
```

---

## 🔧 Method Adaptation Details

### 🧪 scVAEIT
- Concatenates lab tests and metabolomics into two Gaussian observation blocks  
- Training: Paired training visits only  
- Inference: Masks metabolomics block; reconstructs solely from lab block  
- ✅ Directly aligns with scVAEIT's masking-based conditional imputation design  

### 🧬 MIDAS
- Treats lab tests and metabolomics as two modalities in `MuData`  
- **Critical adaptation**: Registers Gaussian decoder at runtime (public MIDAS ships only with Poisson/Bernoulli decoders)  
- All samples assigned identical batch label (batch correction not targeted in this benchmark)  
- Inference: Runs on lab-only query `MuData` using trained checkpoint  

### 🌐 StabMap
- **Reference**: Training paired visits (shared lab features + unshared metabolomics features)  
- **Query**: Test visits (lab features only)  
- Shared lab features establish mosaic topology between reference and query  
- `imputeEmbedding()` transfers full feature space to query; metabolomics rows extracted as predictions  
- ✅ Closest StabMap analogue to "current lab → current metabolomics" task  

---

## 🚀 Standard Workflow

### 1. Setup external model implementations
```bash
# Create directory structure
mkdir -p baseline-model

# Clone required model repositories
git clone https://github.com/jaydu1/scVAEIT.git baseline-model/scVAEIT
git clone https://github.com/labomics/midas.git baseline-model/midas
git clone https://github.com/MarioniLab/StabMap.git baseline-model/StabMap
```

### 2. Run benchmark pipeline
```bash
# Prepare standardized matrices (required for StabMap; optional for others)
python scripts/prepare_tabular_benchmark.py \
  --input /path/to/split_dir \
  --output-dir ./outputs/prepared

# Run method-specific pipelines
python scripts/run_scvaeit.py --input /path/to/split_dir --output-dir ./outputs/scvaeit
python scripts/run_midas.py --input /path/to/split_dir --output-dir ./outputs/midas
Rscript r/run_stabmap_benchmark.R ./outputs/prepared ./outputs/stabmap

# (Optional) Re-evaluate existing predictions
python scripts/evaluate_predictions.py \
  --truth ./outputs/prepared/test_metab.csv \
  --prediction ./outputs/stabmap/test_metab_pred.csv \
  --output ./outputs/stabmap/metrics.json
```

---

## 📦 Dependencies

### Python
```txt
numpy pandas scikit-learn scipy
anndata mudata torch lightning
tensorflow tensorflow-probability scanpy
```

### R (StabMap)
```r
jsonlite, Matrix, igraph, scater, scran,
BiocNeighbors, abind, slam
```

---

## ℹ️ Critical Notes

1. **External Model Sources**:  
   This repository **does not include** the actual model implementations. You must clone them manually:
   ```bash
   git clone https://github.com/jaydu1/scVAEIT.git baseline-model/scVAEIT
   git clone https://github.com/labomics/midas.git baseline-model/midas
   git clone https://github.com/MarioniLab/StabMap.git baseline-model/StabMap
   ```

2. **Design Philosophy**:  
   - All adaptation logic explicitly documented for reproducibility and review transparency  
   - Strict adherence to "no historical context" static protocol  
   - Metabolomics imputation uses **only current-visit lab data**  

3. **Scope Clarification**:  
   This benchmark focuses on evaluating static multimodal imputation capabilities in clinical settings. For longitudinal modeling, consider methods specifically designed for temporal dynamics.

---

> 🌐 This benchmark provides a clear, reproducible framework for comparing static multimodal imputation methods in clinical contexts.  
> 📬 Report issues or contribute improvements via GitHub Issues.