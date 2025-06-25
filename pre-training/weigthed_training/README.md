# ComplEx weighted training

This directory contains a files for manipulating the loss function based on how each triple is likely to occur based on the recommendations from the server. 

### If you combine the triples extension functionality too, then firstly run the recommender system. e.g.
1. For outgoing relations: 
./RecommenderServer serve data/FB15k237_outgoing.tsv.schemaTree.typed.pb
2. For both outgoing and incoming relations: 
./RecommenderServer serve data/FB15k237_bidirectional.tsv.schemaTree.typed.pb
3. For both outgoing and incoming relations with their types: 
./RecommenderServer serve data/FB15k237_bidirectional_typed.tsv.schemaTree.typed.pb

### Run Weighted ComplEx 
```bash
python complex_weighted_training_pipeline.py
```
