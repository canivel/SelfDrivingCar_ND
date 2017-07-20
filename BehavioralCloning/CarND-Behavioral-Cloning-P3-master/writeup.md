**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
How I adressed each point of my implementation:
* Research, find out what is the best approaches been used for this kind a problem
* Create the dataset, using the simulator to record multiples runs, in two different tracks, makign some mistakes and recovery to no be bias at the training.
* Implemente augmentation to generalize the model
* Create the model based on the researched models
* Training the model with different hyperparameters
* Analize the training based on the loss and mean square error
* Test the model with the simulator


---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is based on the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

The architecture seems to be best one for this project

The following additions were made.
- A lambda layer (normalization) is added.
- A crop layer is added.
- The model was normalised, which help to avoid saturation and gradients work better.
- A specific learning rate is used for the adam optimizer.
- In order to avoid overfitting a dropout layer has been added.

|layer				 | shape  				 |
|:------------------:|:---------------------:|
|Input 160, 320, 3  |
|Lambda          | (lambda x: x / 127.5 - 1.)|
|Cropping        | (50,20)
|Convolution 		 | Filter 24, Kernel (5 x 5), Stride (2 x 2) , "relu"|
|Convolution 		 | Filter 36, Kernel (5 x 5), Stride (2 x 2) , "relu"|
|Convolution 		 | Filter 48, Kernel (5 x 5), Stride (2 x 2) , "relu"|
|Convolution 		 | Filter 64, Kernel (3 x 3), Stride (2 x 2) , "relu"|
|Convolution 		 | Filter 64, Kernel (3 x 3), Stride (2 x 2) , "relu"|
|Dropout 		 	 | 0.5					 |	
|Flatten 		 	 | 
|Dense  		 	 | 100, "relu"			 | 
|Dense  		 	 | 50, "relu"			 |
|Dense  		 	 | 10, "relu"			 |
|Dense  		 	 | 1 			 		 |


The model uses RELU activations functions. 
The impletation was built using  Keras

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

####3. Model parameter tuning

The model used an adam optimizer, with LEARNING_RATE = 1e-4

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

The data was generated using the Udacity simulator, in both tracks, using the keyboard. I record at least 5 times of full laps in both tracks.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the examples in teh past videos, but with the augmented data, which fail baddly.

Then I tried to implement the comma.ai model, which took me to the Nvidia model

My first try with the Nvidia model was without dropouts, which generated a higher loss on the validation set when compared to the training set. I added a droput layer to fix the overfitting issue.

Then I found some issues in recovery from fails, so I added a correction factor of 0.2 to the angles, like I found in many examples around the web, this increased the perspective of the camera.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

###Problems to be solved
1)Speed control, need to be able to learn better speed control
2)At the second track at downhill the breakes go crazy, it need a better training data for breaking scerarios too.
