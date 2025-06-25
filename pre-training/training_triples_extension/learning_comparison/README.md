# ComplEx Training Comparison with PyKEEN Callbacks

This directory contains a files to compare baseline and bidirectional ComplEx models using PyKEEN's built-in callback system.

It uses pipeline() function, built-in callback system, exact parameter matching, and logging

### Firstly run the recommender system. e.g.
./RecommenderServer serve data/FB15k237_bidirectional.tsv.schemaTree.typed.pb

### Run Complete Comparison (Recommended)
```bash
# Run both trainings + plots (5 epochs for testing)
python run_comparison.py --max-epochs 5

# Run full training (100 epochs like original)
python run_comparison.py --max-epochs 100
```

### Run Individual Models
```bash
# Train baseline only
python complex_baseline_with_callbacks.py --max-epochs 5

# Train bidirectional only  
python complex_bidirectional_with_callbacks.py --max-epochs 5 --max-entities 100

# Generate plots from existing CSV files
python plot_training_metrics.py
```

## **Output Structure**

```
models/
├── baseline_complex_with_callbacks/
│   ├── baseline_epoch_metrics.csv        # Per-epoch metrics
│   ├── baseline_epoch_metrics.json       # JSON format
│   ├── baseline_trained_model.pkl        # Trained model
│   └── final_metrics.txt                 # Final test results
├── bidirectional_complex_with_callbacks/
│   ├── bidirectional_epoch_metrics.csv   # Per-epoch metrics
│   ├── bidirectional_epoch_metrics.json  # JSON format
│   ├── bidirectional_trained_model.pkl   # Trained model
│   └── final_metrics.txt                 # Final test results
└── metrics_comparison_plots/
    ├── training_metrics_comparison.png    # 6-panel comparison
    ├── training_metrics_comparison.pdf    # PDF version
    ├── hits_at_1_comparison.png          # Individual plots
    ├── hits_at_3_comparison.png
    ├── hits_at_5_comparison.png
    ├── hits_at_10_comparison.png
    ├── mrr_comparison.png
    ├── mean_rank_comparison.png
    └── training_summary.txt               # Detailed comparison
```

## **Command Line Options**

### `run_comparison.py`
```bash
--max-epochs 5              # Training epochs (default: 5 for testing)
--max-entities 100          # Entities for bidirectional (default: 100)  
--probability-threshold 0.5  # Recommendation threshold (default: 0.5)
--baseline-output DIR        # Baseline output directory
--bidirectional-output DIR   # Bidirectional output directory
--plots-output DIR          # Plots output directory
```

## **Example Output**

The comparison plots show training curves like:

```
ComplEx Training Metrics: Baseline vs. Bidirectional

[Hits@1]    [Hits@3]    [Hits@5]
[Hits@10]   [MRR]       [Mean Rank]
```

Each plot shows:
- **Blue circles**: Baseline model  
- **Orange squares**: Bidirectional model

## **Testing vs Production**


## 🔍 **Troubleshooting**

**API Connection Issues:**
- Ensure recommender API is running on `localhost:8080`
- Check `get_config('api.url')` in bidirectional script

**Memory Issues:**
- Reduce `--max-entities` parameter
- Reduce `--max-epochs` for testing

**Plotting Issues:**
- Install: `pip install matplotlib seaborn pandas`
- Check CSV files exist before plotting
