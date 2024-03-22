# UMF: Unifying Local and Global Multimodal Features
![alt text](figures/UMF_architecture.png "UMF architecture")


### Abstract

Perceptual aliasing and weak textures pose significant challenges to the task of place recognition, hindering the performance of Simultaneous Localization and Mapping (SLAM) systems. This paper presents a novel model, called UMF (standing for Unifying Local and Global Multimodal Features) that 1) leverages multi-modality by cross-attention blocks between vision and LiDAR features, and 2)includes a re-ranking stage that re-orders based on local feature matching the top-k candidates retrieved using a global representation. Our experiments, particularly on sequences captured on a planetary-analogous environment, show that UMF outperforms significantly previous baselines in those challenging aliased environments. Since our work aims to enhance the reliability of SLAM in all situations, we also explore its performance on the widely used RobotCar dataset, for broader applicability.

### Citation
If you find our work useful, please cite us:
```
@article{garcíahernández2024unifying,
      title={Unifying Local and Global Multimodal Features for Place Recognition in Aliased and Low-Texture Environments}, 
      author={Alberto García-Hernández and Riccardo Giubilato and Klaus H. Strobl and Javier Civera and Rudolph Triebel},
      year={2024},
      eprint={2403.13395},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


### Environment and Dependencies

Code was tested using Python 3.8 with PyTorch 1.9.1 on Ubuntu 20.04 with CUDA 10.6.

The following Python packages are required:
* PyTorch
* pytorch_metric_learning (version 1.0 or above)
* spconv
* einops
* opencv-python



*  install [HOW](https://github.com/gtolias/how)
```
git clone https://github.com/gtolias/how
export PYTHONPATH=${PYTHONPATH}:$(realpath how)
```

*  install [cirtorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch/)
```
wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/v1.2.zip"
unzip v1.2.zip
rm v1.2.zip
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval)
```

Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/.../UMF
export PYTHONPATH=$PYTHONPATH:/home_local/$USER/UMF
export PYTHONPATH=$PYTHONPATH:/home/.../UMF
export PYTHONPATH=$PYTHONPATH:/home_local/$USER/UMF
export PYTHONPATH=${PYTHONPATH}:$(realpath how)
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval)
```



### Training

To train UMF following the procedure in the paper first download the Robotcar dataset. Otherwise, adapt the dataloader accordingly.

Edit the configuration files:
- `config_base.yaml`      # select the etna or robotcar version
- `./models/UMF/UMFnet.yml`            # fusion model only
- `./models/UMF/UMFnet_ransac.yml`     # multimodal with ransac reranking 
- `./models/UMF/UMFnet_superfeat.yml`  # multimodal with superfeatures reranking 

Modify `batch_size` parameter depending on the available GPU memory. 


Set `dataset_folder` parameter to the dataset root folder, where 3D point clouds are located.
Set `image_path ` parameter to the path with RGB images corresponding to 3D point clouds, extracted from 


To train, run:

```train 
# Fusion only
python train.py --config ../config/config_base.yaml --model_config ../models/UMFnet.yml
# RANSAC
python train.py --config ../config/config_base.yaml --model_config ../models/UMFnet_ransac.yml
```
We provide the pre-trained models for the Robotcar datatset ([link](https://drive.google.com/drive/folders/1MXOhMC6wxjU0FjsDM1GzUIzJJ0e-5mjQ?usp=sharing)).


### Evaluation

To evaluate pretrained models run the following commands:

```
cd eval

# Evaluate with superfeatures or ransac variant
python evaluate.py --config ../config/config_base.yaml --model_config ../models/UMFnet_ransac.yml --weights <path_to_weights>

```
