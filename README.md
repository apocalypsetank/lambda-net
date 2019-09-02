# Lambda-net



This repository contains the coder for the paper: **λ-net: Reconstruct Hyperspectral Images from a Snapshot Measurement** (***International Conference on Computer Vision*** 2019) by Xin Miao, Xin Yuan, Yunchen Pu, Vassilis Athitsos.





## Training and testing data.

The traing and testing data used in our paper is from the paper " Sparse Recovery of Hyperspectral Signal from Natural RGB Images". It can be downlaed from http://icvl.cs.bgu.ac.il/hyperspectral/.  The training scene should be put in the file "training_data/",the testing scenes should be in the file "testing_data/". The data we used and the mask can be downloaded from https://drive.google.com/open?id=1lJB9Ekif4fXQSL9fGzrsxgp5YFMb7FEu.


## Usage
### 0. Download the DeSCI repository

Requirements are tensorflow, numpy, scipy.

### 1. Reconstruction Stage

Train the model via
```
main.py
```
The results for testing scene will be saved in 'result/your model file/'. If you want to visualize the results after training the model, just run
```
visualize_results.py
```

