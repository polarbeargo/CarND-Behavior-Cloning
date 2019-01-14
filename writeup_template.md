# Behavioral Cloning #

The goal of this project is to clone human driving behavior using a Deep Neural Network to achieve this goal, I use a simple Car Simulator provide by Udacity.



  
  The steps of this project are the following:  
  
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left_2017_09_02_21_56_50_551.jpg 
[image2]: ./examples/center_2017_09_02_21_56_50_551.jpg 
[image3]: ./examples/right_2017_09_02_21_56_50_551.jpg 
[image4]: ./examples/left_2017_09_02_21_57_46_704.jpg 
[image5]: ./examples/center_2017_09_02_21_57_46_704.jpg 
[image6]: ./examples/right_2017_09_02_21_57_46_704.jpg 
[image7]: ./examples/left_2017_09_02_21_58_05_226.jpg 
[image8]: ./examples/center_2017_09_02_21_58_05_226.jpg 
[image9]: ./examples/right_2017_09_02_21_58_05_226.jpg 
[image10]: ./examples/nVidia_model.png 
[image11]: ./examples/center-2017-02-06-16-20-04-855.jpg 
[image12]: ./examples/center-2017-02-06-16-20-04-855-flipped.jpg 
[image13]: ./examples/original-image.jpg 
[image14]: ./examples/cropped-image.jpg 
[image15]: ./examples/suffled.png
[image16]: ./examples/lakeDataFrame.png
[image17]: ./examples/lakeSteering.png
[image18]: ./examples/valley.png
[image19]: ./examples/valleycorrection.png


## Exploring the Data. ##
The simulator captures images from three cameras mounted on the car: a center, right and left camera to recovering from being off-center. Track one dataset was preprocessed and trained on 29750 sample, validated on 7438 sample and Track two was preprocess and trained on 8160 samples, validated on 2040 samples .


| Left          | Center        | Right  |
| :-------------: |:-------------:| :------:|
|![alt text][image1] | ![alt text][image2] | ![alt text][image3] |
|![alt text][image4] | ![alt text][image5] | ![alt text][image6] |
|![alt text][image7] | ![alt text][image8] | ![alt text][image9] |

![][image16]

### Files Submitted & Code Quality ###

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode ####
  
  
My project includes the following files:  

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* lakesucvelocity25.mp4 and v3.mp4 vedio file for demonstrate the result.


#### 2. Submission includes functional code ####
Using the Udacity provided simulator and my drive.py file which has modeified the speed to 25MPH(9MPH in track 2), the car can be driven autonomously around the track one by executing 
```sh
python drive.py modelsucesslake.h5
            or
python drive.py modellakesucveloci25.h5
            or
python drive.py model.h5(For second submission)
            
```

#### 3. Submission code is usable and readable ####

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy ###

#### 1. An appropriate model architecture has been employed ####

My convolutional neural network architecture was inspired by NVIDIA's End to End Learning for Self-Driving Cars paper. The original Nvidia has a tenserflow version, I transfered it into Keras environment. The main difference between my model and the NVIDIA mode is than I did use drpout layers. My model consists of a convolution neural network with 3x3 filter sizes and 5 convolutional layers each one has 24 to 64 filters, strid of 2 or 1 and 5 fully connected layers(`model.py` line [70 to 83](model.py#L70-L83) ). There are 2,712,951 total/trainable parameters came out of this model.
The 5 convolutional layer shrink down layer by layer, because I am using the 'valid' padding, until it reach 64 then flatten the 64 layers down, get 64x1x33 = 2112 connectors. Then follow by 5 fully connected layers. The first dense layer has 1164 connectors, therefore, between flatten layer and dense_1 layer has 2112x1164+1164 = 2459532 connections. 

The model includes RELU layers to introduce nonlinearity (code line 70-81), and the data is normalized in the model using a Keras lambda layer, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py. And divide the pixel by 127.5 which gave a much better result than divide by 255 from try and error steps then minus 1, yield new value between (-1 to 1) (code line 68). 
<img src="./examples/nVidia_model.png?raw=true" width="600px">

```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 157, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 77, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          2459532     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dropout_2[0][0]                  
====================================================================================================
Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0
____________________________________________________________________________________________________

```
#### 2. Attempts to reduce overfitting in the model ####

The model contains dropout layers in order to reduce overfitting (model.py lines 78, lines 82). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 66). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning ####

The model used Adam optimiser to control learning rate = 1e-04 (model.py line 84).

#### 4. Appropriate training data ####

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

#### 5. Creation of the Training Set & Training Process ####

To capture good driving behavior, I first recorded two laps( both clock and counterclockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road. These images show what a recovery looks like starting from ... :

| Left          | Center        | Right  |
| :-------------: |:-------------:| :------:|
|![alt text][image1] | ![alt text][image2] | ![alt text][image3] |

#### Track one ####

![alt text][image17]
![alt text][image15]

#### Track two ####

![alt text][image19]
![alt text][image18]


Then I repeated this process on track two in order to get more data points.
To augment the data sat, I also flipped images and angles thinking that this would helps my model generalize better. As suggested by Udacity, driving in opposite direction also helps my model. The reason is the lap has too many left turns. By driving in reversed direction, to force my model to learn the right turn too.For example, here is an image that has then been flipped:

| Image captured by the center camera          | Flipped image from the center camera        | 
| :-------------: |:-------------:|
|![alt text][image11] | ![alt text][image12] |

The Cropping2D layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.

Here is an example of an input image and its cropped version after passing through a Cropping2D layer:

| Original image taken from the simulator | Cropped image after passing through a Cropping2D layer | 
| :-------------------------------------: |:------------------------------------------------------:|
|![alt text][image13]                     |          ![alt text][image14] |

After the collection process, I then preprocessed this data by shuffle the training data before each epoch, shuffled the data set and put 20% of the data into a validation set. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by try and error approach, I used an adam optimizer the learning rate = 1e-04.

#### 4.Result and Recommendation for Improvements ####

In the initial stage of the project, I used a dataset generated by myself. That dataset was small and recorded while navigating the car using the laptop keyboard and adding steering correction from 0.1-0.3. However, the model built was not good enough to autonomously navigate the car in the simulator. I think while driving the car and keep the car on the center of the road, adding any correction might not help. It's just get worse and can't make one lap in both track so I decide to remove the steering correction part and keep the model simple. After tons of try out in track 2 like collecting more data and keep the car in the middle of the road or collected clock and counterclockwise loop or corrected steering angle, I noticed that maybe adjust Brightness of image data will be way more helpful than only correcting steering angle. After the reviwer's advice I able to correct the following steps:  

* Aligning the color spaces used by drive.py and model.py. Currently it looks like your model.py uses cv2 to read images in BGR while drive.py uses PIL to read images in RGB.
* Augmenting the training data with image processing techniques to create more diverse examples for the model to learn from. I adjust the brightness of a image, to make new image look like under a shade.  

**Because of my training model data was collected under screen resolution 640 X 480 whith graphic quality fastest (because I have problem to complete one loop under high resolution mode..) please run the simulater in the same condition, that way the car can completed many laps autonomously many times.**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ufS0aC3DW6c/0.jpg)](https://www.youtube.com/watch?v=ufS0aC3DW6c)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/FcSmC20iiTI/0.jpg)](https://www.youtube.com/watch?v=FcSmC20iiTI&feature=youtu.be)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/iENojwOzI3U/0.jpg)](https://www.youtube.com/watch?v=iENojwOzI3U&feature=youtu.be)
