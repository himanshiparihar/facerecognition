## USAGE

1. `train.py`: Classification file for training and inference.  
	1. For using trained model:  
	`python train.py hymenoptera_data2/ 0`  
	2. For training:  
	`python train.py hymenoptera_data2/ 1`  

2. `readVid.py`: For streaming images from mobile camera to opencv, using `droidcam` app for Android.  
	1. `python readVid.py`  

## TODO

1. Put hyperparameters to separate python file.  
2. Use `argparse` python.  


## References 

1. [DroidCam](http://ubuntuhandbook.org/index.php/2015/01/use-android-phone-as-wireless-webcam/)  
2. [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)  
3. [Medium Blog on Transfer Learning](https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5)  
4. [Medium Blog: ResNet OOPs implementation in Pytorch](https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)  
5. [Medium Blog: Layer wise explanation of ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)  
6. Custom loss function in Pytorch: [Pytorch forum](https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235) and [basic examples](https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/).  
