# CIFAR_10-with-pytorch

## 1. Introduction

A image classification demo project on CIFAR-10 dataset using PyTorch. Currently this project implemented two models, which are LeNet-5 and VGG-16. As for VGG-16, the test accuracy can reach 84%，and the running state of the program is shown as below:
```text
PS D:\Files\Github\CIFAR_10-with-pytorch> python .\main.py --do_eval --vgg
Start checking path...
Check path done.
Files already downloaded and verified
Files already downloaded and verified
Testing...
Test Iteration: 100%|█████████████| 2500/2500 [01:44<00:00, 24.02it/s]
10000张测试集中的准确率为: 84 %
```
I also implemented a simple GUI using streamlit:
![demo](demo.jpg)

## 2. Dependencies

* pytorch
* tqdm
* argparse
* streamlit


## 3. Directories

Due to the size limitation of uploaded files on Github, some large test images and trained model parameter files are not uploaded. Below is my local project directory structure, you can refer and modify as needed.

```text
root/
  |_ data/
  |   |_ CIFAR-10 original data.
  |
  |_ model/
  |   |_ Well-trained model prameter files.
  |
  |_ test/
  |   |_ Some test images.
  |
  |_ web/
  |   |_ app.py 
  |
  |_ __init__.py
  |
  |_ dataset.py
  |_ main.py
  |_ model.py
  |_ predict.py
  |_ test.py
  |_ train.py
  |_ unil.py
  |
  |_ README.md
  |_ .gitignore
```

Some Hints:

* `data/` directory is used for storing the original CIFAR-10 dataset
* `test/` directory is used for storing some custom test images
* `web/` directory implements a simple GUI based on streamlit
* `dataset.py` is used for accepting input data from CIFAR-10
* `model.py` implements two neural network models, which are LeNet-5 and VGG-16. You can implement any other 


## 4. Train

```shell
python ./main.py --do_train [--vgg/--lenet]
```


## 5. Test

```shell
python ./main.py --do_eval [--vgg/--lenet]
```


## 6. Predict

```shell
python ./main.py --do_predict [--vgg/--lenet]
```


## 7. Run web application

```shell
cd web
streamlit run app.py
```