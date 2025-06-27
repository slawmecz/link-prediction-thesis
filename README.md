
# Enhancing Link Prediction in Knowledge Graphs Through Pre-Informed Training

This repository is a part of a research implementation conducted by Slawomir Meczynski (s.a.meczynski@student.vu.nl). The project was completed under the supervision of Daniel Daza and Michael Cochez as a bachelor's thesis. The full research paper is available as a PDF in the root directory.

## Repository Structure

- `recommender_server/`  
  Core system that returns likely to be queried properties for a given entity.

- `data_configuration/`  
  Utility files for retrieving the type information of entities.

- `pre_training/`  
  Contains two strategies for "pre-informing" the model about likely queried properties. It also includes trained models and statistical testing scripts.

- `recommender_experiments/`  
  Contains experiments run on the dataset FB15k237 in order to check the functionalities of the recommendations system.

## Model & Dataset

- **Baseline Model:**  
  [ComplEx](https://proceedings.mlr.press/v48/trouillon16.html)  
  > Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. *International Conference on Machine Learning*, pp. 2071–2080. PMLR.

- **Dataset:**  
  [FB15k-237](https://paperswithcode.com/dataset/fb15k-237)  
  > Toutanova, K., & Chen, D. (2015). Observed versus latent features for knowledge base and text inference.
 
The implementation also supports datasets in Wikidata format, such as [CoDEx](https://github.com/tsafavi/codex) (already included in the pre-training files), but can be run with other knowledge graphs as well. The files used to generate the TSV required for building the SchemaTree are located in the recommender_server directory. The instructions on how to build and use the SchemaTree are in the README file of the recommender_server directory.



