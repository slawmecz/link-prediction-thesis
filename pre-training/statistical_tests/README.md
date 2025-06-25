# ComplEx Statistical Comparison between the baseline model and extended

This directory contains a files for statistical comparison.

### Firstly run the recommender system. e.g.
./RecommenderServer serve data/FB15k237_bidirectional.tsv.schemaTree.typed.pb

### Run Complete Significance Test (Recommended)
```bash
python run_significance_test.py --baseline-dir models/baseline --extended-dir models/extended
```



