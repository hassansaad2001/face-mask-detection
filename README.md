# face mask detection with in python with pytorch

## introduction
  this project is face mask detction with pytorch ,i'll extract faces from image(frame from camera) by pretrained model(caffe) and this faces is pass to our model to predict if a person is wearing face mask or not.
  green rectangle with ask and blue rectangle without mask.
## requirements libraries
  1- pytorch
  2- PIL
  3- numpy
  4- matplotlib
  5- sklearn
  6- seaborn
  7- pandas
  8- opencv
  9- os
## dataset  
  3 directory,train,validation and test

## training  
  1- preprocessing  
      image transformations(Resize,ToTensor,Normalize,RandomHorizontalFlip,RandomVerticalFlip)
      loading data from location and apply transformations
      data to batchs
      
  2- loading model  
      i used pretrained model mobilenet_v2
      
  3- fine-tuning  
      freezing the initial layers of MobileNetv2(train only classifier layer)
      add classifier layer(last layer)
      
  4- choosing optimizer and loss Function  
      Adam and CrossEntropyLoss 
      
  5- training  
      1- train model on only last layer and freezing initial layers (acc:0.9225  ,loss:0.3886,lr=0.0001 ,Epoch:25)  
      2- train model on last layer and (10-18 initial layers) (acc:0.9950  ,loss:0.0.3183,lr=0.0001,Epoch:25)  
      3- train model on all layers(acc:0.1.0000,loss:0.0.3133 ,lr=0.00001,Epoch:10)  
  6- compute cv accuracy and plot loss,acc  

## testing  
  accuracy = 0.997984  
  
  ![cv_acc_test](https://user-images.githubusercontent.com/90579377/201539906-22aad5c2-da7e-4edf-97e5-66aa6015a3f1.png)  
  ![plot_acc](https://user-images.githubusercontent.com/90579377/201539919-7c358a5d-7924-407c-a969-1063ff09b109.png)  
  ![plot_loss](https://user-images.githubusercontent.com/90579377/201539928-4ee7c89d-05e2-4e39-a1bd-ec117a5f8459.png)  

## usage  
  run python file 'runing_with_camera.py'




  





