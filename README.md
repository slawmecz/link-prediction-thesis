
# Enhancing Link Prediction in Knowledge Graphs Through Pre-Informed Training

This repository is a part of an implementation of research conducted by Slawomir Meczynski (s.a.meczynski@student.vu.nl). The project was done under supervision of Daniel Daza and Michael Cochez as a bachelor thesis of Slawomir Meczynski. The full research paper is available as a PDF in the root directory.

## Repository Structure

- `recommender_server/`  
  Core system that returns likely to be queried properties for a given entity.

- `data_configuration/`  
  Utility files for retrieving the type information of entities.

- `pre_training/`  
  Contains two strategies for "pre-informing" the model about likely queried properties.

- `recommender_experiments/`  
  Contains experiments run on the dataset FB15k237.

## Model & Dataset

- **Baseline Model:**  
  [ComplEx](https://proceedings.mlr.press/v48/trouillon16.html)  
  > Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. *International Conference on Machine Learning*, pp. 2071–2080. PMLR.

- **Dataset:**  
  [FB15k-237](https://paperswithcode.com/dataset/fb15k-237)  
  > Toutanova, K., & Chen, D. (2015). Observed versus latent features for knowledge base and text inference.



