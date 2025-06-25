# ComplEx training set extension

This directory contains a files for training the baseline ComplEx as well as extending the training set. This involves several approaches - adding only outgoing relations, both incoming and outgoing, and typed (including the type of a given entity). 

### Firstly run the recommender system. e.g.
1. For outgoing relations: 
./RecommenderServer serve data/FB15k237_outgoing.tsv.schemaTree.typed.pb
2. For both outgoing and incoming relations: 
./RecommenderServer serve data/FB15k237_bidirectional.tsv.schemaTree.typed.pb
3. For both outgoing and incoming relations with their types: 
./RecommenderServer serve data/FB15k237_bidirectional_typed.tsv.schemaTree.typed.pb

### Run Baseline ComplEx 
```bash
python complex_baseline.py \
    --output-dir models/my_baseline \
    --dataset FB15k237 \
    --embedding-dim 1000 \
    --max-epochs 19
```
### Run Outgoing ComplEx (Recommended)
```bash
python complex_extended_bidirectional.py \
    python complex_extended_outgoing_correct.py \
    --output-dir models/extended_outgoing \
    --baseline-model-dir models/baseline_complex \
    --dataset FB15k237 \
    --model ComplEx \
    --embedding-dim 1000 \
    --max-epochs 100 \
    --probability-threshold 0.25 \
    --sampling-rate 0.0 \
    --api-url http://localhost:8080/recommender
```
### Run Bidirectional ComplEx (Recommended)
```bash
python complex_extended_bidirectional.py \
    --output-dir models/extended_outgoing \
    --baseline-model-dir models/baseline_complex \
    --dataset FB15k237 \
    --model ComplEx \
    --embedding-dim 1000 \
    --max-epochs 100 \
    --probability-threshold 0.25 \
    --sampling-rate 0.0 \
    --api-url http://localhost:8080/recommender
```

### Run Bidirectional Typed ComplEx (Recommended)
```bash
python complex_extended_bidirectional_typed.py \
    --output-dir models/extended_outgoing \
    --baseline-model-dir models/baseline_complex \
    --dataset FB15k237 \
    --model ComplEx \
    --embedding-dim 1000 \
    --max-epochs 100 \
    --probability-threshold 0.25 \
    --sampling-rate 0.0 \
    --api-url http://localhost:8080/recommender
```



