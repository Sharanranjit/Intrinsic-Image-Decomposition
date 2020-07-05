# Intrinsic-Image-Decomposition
The Official implementation of the [Single Image Intrinsic Decomposition Using Transfer Learning](https://dl.acm.org/doi/abs/10.1145/3383972.3384062)  
Authors: [Sharan Ranjit](https://github.com/Sharanranjit/), Dr.Raj K Jaiswal  
**Publication: ICMLC 2020: Proceedings of the 2020 12th International Conference on Machine Learning and Computing, February 2020** 

## What is intrinsic image decomposition ?
Every RGB image can be assumed to be composed of two maps, which are albedo and shading. In simple words, separating an RGB image into these two maps describes the idea behind intrinsic decomposition.

![alt text](https://github.com/Sharanranjit/Intrinsic-Image-Decomposition/blob/master/demo.png "Demonstration of IID")

## Data
We use the [MPi-Sintel](http://sintel.is.tue.mpg.de/) dataset and generate 8900 images. 
* Training data(4 GB): [MPI.zip](https://drive.google.com/uc?export=download&id=1t2ZanrGGeic1BEeXZoyLFzDz6BAD3Dfo)
* Test data(593 MB): [MPI_2.zip](https://drive.google.com/uc?export=download&id=1e2UlScEz3LZzePEcDZlEHftVEcm4utWa) 

The zip files must be in the same directory and no extraction is required. 

## Implementation 
The code is based on **Keras 2.3.1**, **Tensorflow 1.15.2**, and **Python 3.6**.The model is trained in **Google Colab** workspace. Code based on Tensorflow 2 coming soon!

### Training
For training on MPI-Sintel dataset, run in terminal: ``` python train.py --epochs 15 ```  
We train the model for 15 epochs with a learning rate of 0.0001 and batch size of 8 images. Approximately, it takes 4hrs to train the model. To modify other parameters, please check the [training](https://github.com/Sharanranjit/Intrinsic-Image-Decomposition/blob/master/train.py) script.

### Testing
For testing, run: ``` python test.py --model 'Models/SIID_121.h5' --input 'Examples/input.png' ``` 
The trained models can be downloaded from below links: 
* [SIID_121.h5](https://drive.google.com/uc?export=download&id=1-U3l4MORPxfeIHLwW-fp2-fgHBjU44nw) => DenseNet-121 as encoder 
* [SIID_169.h5](https://drive.google.com/uc?export=download&id=1-dCJm0m5mz91w2n3kNVQQDEkOnxHtwwF) => DenseNet-169 as encoder

**Note: The model currently accepts images whose height and width are a multiple of 32. You can resize the image 
to the nearest multiple.**

## Sample outputs 
![alt text](https://github.com/Sharanranjit/Intrinsic-Image-Decomposition/blob/master/Examples/comb_outs.png "Sample outputs")

## Reference
``` 
@inproceedings{10.1145/3383972.3384062,
author = {Ranjit, S Sharan and Jaiswal, Raj K.},
title = {Single Image Intrinsic Decomposition Using Transfer Learning},
year = {2020},
isbn = {9781450376426},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3383972.3384062},
doi = {10.1145/3383972.3384062},
booktitle = {Proceedings of the 2020 12th International Conference on Machine Learning and Computing},
pages = {418–425},
numpages = {8},
keywords = {Shading Image, Albedo Image, Transfer Learning, Intrinsic decomposition, CNN},
location = {Shenzhen, China},
series = {ICMLC 2020}
}
```  
