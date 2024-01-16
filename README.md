# Topology optimization with deep learning

[SIMP_multi.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/simp/SIMP_multi.py) code used for generate training dataset.

[CNN.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/CNN.py) code used for training neural network.

[load_model.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/CNN2.py) code used for load neural network.

[SIMP_multi_dist.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/SIMP_multi_dist.py) code used for generate dataset with a distributed load.

### Results folders

- `results_1f` Folder with the data with just 1 force
- `results_2f` Folder with the data with just 2 force is divided in third parts, this contain the first part with the first columns.
- `results_2f_2` Folder with the data with just 2 force second part
- `results_2f_3` Folder with the data with just 2 force third part
- `results_merge_#` Folder with the data mix of all folders
- `results_merge_3` Folder with the data mix of all 2f results 2f_1 and 2f_3
- `results_dist` Folder with the data of distribution loads
- `results_rand` Folder with the data random 10000
- `results_rand4` Folder with the data random 20000 with just vertical forces
- `results_rand5` Folder with the data random 40000 with just vertical forces
- `results_rand_6` Folder with the data random 40000 with just vertical forces
- `results_rand_7` Folder with the data random 40000 with just vertical forces and no volumen change 0.5 for all


### Model folder

- `Basic_NN` Model more basic, don't work at all
- `first_NN` Model with more data
- `second_NN` Model with data of 2 loads
- `third_NN` Model with data mix of merge results
- `U_NN` U-Net model with data mix of merge results
- `U_NN2` U-Net model with data mix of merge results 50 epoch
- `ViT_test` ViT model with data mix of merge results 50 epoch wrong data
- `ViT2` ViT model with data mix of merge 2 results 50 epoch
- `ViT3` ViT model with more parameters 8M approx 50 epoch
- `vit_last_100` ViT model with 100 epoch batch size small
- `unn_last_100` U-Net model with 100 epoch batch size small
- `model_unet_merge` U-Net model with earlyStop 5 epochs and train with results_merge_3
- `model_rand_#` U-Net model with 50 epochs the # correspond to the number of the rand dataset