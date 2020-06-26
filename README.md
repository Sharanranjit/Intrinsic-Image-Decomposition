# Intrinsic-Image-Decomposition
The Official implementation of the paper [Single Image Intrinsic Decomposition Using Transfer Learning](https://dl.acm.org/doi/abs/10.1145/3383972.3384062)  
Authors: [Sharan Ranjit](https://github.com/Sharanranjit/), Dr.Raj K Jaiswal  
**Publication: ICMLC 2020: Proceedings of the 2020 12th International Conference on Machine Learning and Computing, February 2020** 

## What is intrinsic image decomposition ?
Every RGB image can be assumed to be composed of two maps, which are albedo and shading. In simple words, separating an RGB image into these two maps describes the idea behind intrinsic decomposition.

![alt text](https://github.com/Sharanranjit/Intrinsic-Image-Decomposition/blob/master/demo.png "Demonstration of IID")

## Sample outputs
## Implementation
! Note: The code is based on **Keras 2.3.1** and **Tensorflow 1.15.2**, and the model is trained in **Google Colab** workspace.
### Training
For training on MPI-Sintel dataset, run in terminal: ``` python train.py --epochs 15 ```  
We train the model for 15 epochs with a learning rate of 0.0001 and batch size of 8 images. To modify other parameters, include ``` --help ``` flag.
### Testing
For testing on different images, run: ``` python test.py --model --input```


 
