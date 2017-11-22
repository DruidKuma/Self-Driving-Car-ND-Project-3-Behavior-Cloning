#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/sample_img.jpg "Model Visualization"
[image2]: ./images/sample_recover.jpg "Starting point for left-right recover"
[image3]: ./images/sample_flip_recover_orig.jpg "Sample Image before flipping"
[image4]: ./images/sample_flip_recover.jpg "Sample Image after flipping"
[image5]: ./images/sample_preprocess.png "Image after preprocessing"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to load and preprocess data, create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing weights for the trained convolution neural network
* model.json containing config for the trained model
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is a simple implementation of CNN in Keras (model.py lines 63-69). 
It contains one convolutional layer with 3x3 filter (model.py line 65) with RELU activation to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 64). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 55,60,73). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used predefined data set of images along with populating the set with recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep the code and model as simple as it is possible along with giving satisfactory results.

My first idea was to use a convolution neural network model similar to the Nvidia. I thought this model might be appropriate because it was recommended in the tutorial videos for the project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it uses Dropout.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (e.g the turn immediately after the bridge). To improve the driving behavior in these cases, I generated more data for sharp turns and recover to the center of the road

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Though the model behave almost correctly (sometimes the car in the simulator still "swam" from right to left and vice versa on the road), I still wanted to reduce its complexity (I think that for this particular task using complex architectures with Python generators (to accommodate 160x320 images) was a bit overkill)

####2. Final Model Architecture

The final model architecture (model.py lines 63-69) consisted only of the 7 layers, namely:

* Lambda layer for data normalization
* Convolutional layer with 3x3 filter and valid padding, followed by RELU activation
* Max pooling layer of size 4x4 with valid padding
* Flatten layer
* Dense layer with 1 neuron to produce the steering angle 

Before entering the model, all incoming images are preprocessed. As proprocess part, I've chosen the conversion to HSV and retaining only S channel, and size conversion from 160x320 to 16x32. Such preprocessing gave me possibility to accomodate training data in memory, not to use Python generators, massively reduce the number of parameters to tune and incredibly speed up the learning process along with saving the good performance.
Please, see example of image before/after preprocessing in the next chapter.

####3. Creation of the Training Set & Training Process

I've started with the example data set, provided for this project. It contains all necessary information to capture good driving behavior.
Example of the image from the dataset (center camera)

![][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the center of the road, and support sharp turns. Example of starting image to recover (center camera) :

![][image2]

To augment the data sat, I also flipped images and angles thinking that this would generate more different data for training purposes. For example, here is an image that has then been flipped:

![][image3]
![][image4]

After the collection process, I had approximately 6500 data points. Data preprocessing is described above, here I'll show an example of an image before/after the preprocessing:

![][image1]
![][image5]

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
After testing I came up with ideal number of epochs (15) and batch size of 128. I used an adam optimizer so that manually training the learning rate wasn't necessary.
