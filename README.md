# **Traffic Sign Recognition** 

## Writeup

[//]: # (Image References)

[network]: ./doc/network.png "MyNet (Very similar to LeNet)"
[dataset]: ./doc/dataset-demo.png "Dataset Summary"
[classes]: ./doc/dataset-classes.png "Distribution of samples"
[preproc]: ./doc/after-preproc.png "Preprocesses images"
[undistorted]: ./doc/undistorted.png "Original"
[distorted]: ./doc/distorted.png "Distorted"
[endofnopassing]: ./doc/endofnopassing.png "End of no passing sign"
[featuremaps]: ./doc/featuremaps.png "Feature maps"

[bars1]: ./doc/bars1.png "Top 5 Softmax Probabilities"
[bars2]: ./doc/bars2.png "Top 5 Softmax Probabilities"
[bars3]: ./doc/bars3.png "Top 5 Softmax Probabilities"
[bars4]: ./doc/bars4.png "Top 5 Softmax Probabilities"
[bars5]: ./doc/bars5.png "Top 5 Softmax Probabilities"
[bars6]: ./doc/bars6.png "Top 5 Softmax Probabilities"
[bars7]: ./doc/bars7.png "Top 5 Softmax Probabilities"
[bars8]: ./doc/bars8.png "Top 5 Softmax Probabilities"


[nolimit]: ./images/cut/end-of-all-speed-limits.jpg "End of all speed limits"
[nopassing]: ./images/cut/no-passing-2.jpg "No passing"
[roadwork]: ./images/cut/road-work.jpg "Road work"
[right]: ./images/cut/turn-right-ahead.jpg "Turn right ahead"
[dangerousleft]: ./images/cut/dangerous-turn-to-the-left.jpg "Dangerous turn to the left?"
[myprivilagedroad]: ./images/cut/my-privilaged-road.jpg "Privilaged road"
[my30]: ./images/cut/my-speed-limit-30.jpg "Speed limit: 30km/h"
[mytrafficsignals]: ./images/cut/my-traffic-signals.jpg "Traffic signals"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it and here is a link to my [project code](https://github.com/tomi92/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.

My submission contains a grid of images which includes randomly choosen images from 6 traffic sign types, 16 images for each type.

![Dataset Summary][dataset]

I also provide a bar diagram which shows the distibution of the training samples.

![Distribution of classes][classes]

The distribution is very uneven. I have tried to make it more even by resampling or augmenting smaller classes, but it did'nt result in a noticable accuracy increase.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

My preprocessing pipeline consists of the following steps:

1. Conversion to grayscale: It didn't significantly change the accuracy, but it made it easier to do the normalization.
2. Saturating the intensity values at 1 and 99 percentile.
3. Min/max normalization to the [0, 1] range.
4. Subtraction of its mean from the image, making the values centered around 0 and in the [-1, 1] range.

With a simple min/max normalization I had (approx.) 1% better validation-accuracies than without it.
The percentile-based method gave an additional (approx.) 1% improvement over simple min/max normalization (this method was mentioned in [3]).

![Preprocessed images][preproc]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data already comes in three parts: training, validation and test set.

The sixth code cell of the IPython notebook contains the code for augmenting the data set.

I have tried several methods of data augmentation but none have made the validation accuracy significantly better, some even made it a little bit worse.
However I understood the importance of data augmentation when I first tried the model on some images from the web. 
The accuracy without data augmentation was as low as 20%. 
Maybe because the images had different rotation, position or size than in the original dataset.
Introducing data augmentation boosted the new image accuracy to 75%. (I acquired even newer images, to avoid optimizing for the "new images" set.)

My data augmentation method is as simple as adding 2 distorted versions of each training image to the training set.
Distortion means a random rotation ([-5; -5] degrees), random scale ([0.9; 1.1]) and random translation ([-3; 3] pixels horizontally and vertically).
I read about this method in [3].

After distortion some pixels of the image would remain without data. I filled those pixels with 'edge' padding.

My final training set had 104397 images. My validation set and test set had 4410 and 12630 images.

![Original][undistorted] ![Distorted][distorted]

Original and distorted image side-by-side

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

![MyNet (Very similar to LeNet)][network]

My architecture is a deep convolutional neural network inspired by two existing architectures: one is LeNet[1], and the other is the one in Ciresan's paper[3].
Its number and types of layers come from LeNet, but the relatively huge number of filters in convolutional layers came for Ciresan. Another important property of Ciresan's network is that it is multi-column, but my network contains only a single column. It makes it a little less accurate, but the training and predition is much faster.

The layers:

```
Layer 1: Convolutional (5x5x1x32) Input = 32x32x1. Output = 28x28x32.
Relu activation.
Pooling. Input = 28x28x32. Output = 14x14x32.

Layer 2: Convolutional (5x5x32x64) Input = 14x14x32. Output = 10x10x64.
Relu activation.
Pooling. Input = 10x10x64. Output = 5x5x64.
Flatten. Input = 5x5x64. Output = 1600.
Dropout 0.7.

Layer 3: Fully Connected. Input = 1875. Output = 120.
Relu activation.
Dropout 0.7.

Layer 4: Fully Connected. Input = 120. Output = 84.
Relu activation.
Dropout 0.7.

Layer 5: Fully Connected. Input = 84. Output = 43.
```

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I used AdamOptimizer, a batch size of 128, at most 30 epochs, a learn rate of 0.001.
Another hyperparameter was the dropout rate which was 0.7 at every place where I used it. 
I have tried changing these parameters but it didn't really increase the accuracy.
I saved the model which had the best validation accuracy.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.983 
* test set accuracy of 0.971

I started out by creating an architecture which could clearly overfit the training data. (It converged to 1.0 training-accuracy in a couple of epochs, but the validation accuracy was much lower.
Then I have added regulators until the overfitting was more-or-less eliminated. I added dropout operations between the fully connected layers. I also tried L2 regularization for the weights (in addition to the dropout), but it made the accuracy worse by a tiny amount.
Then I have kept removing filters up to the point when the accuracy started decreasing.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have aquired images from the web and I also made some photos on the streets myself.
I cut out the images similarly to the dataset, but I didn't pay attention to the exact size and position of the signs. I programmatically resized them to 32x32 (not keeping aspect ratio).

###### Photos from the web:

![End of all speed limits][nolimit]

![No passing][nopassing]

This picture is interesting because the perspective and rotation makes the car figures almost form a diagonal similar to the one in End of all speed limits sign.

![Road work][roadwork]

![Turn right ahead][right]

This picture might be hard to classify because it is from a strange perspective and another sign is hanging in to the picture.

![Dangerous turn to the left?][dangerousleft]

This is a very blurry image, at first even I didn't know which sign is it. It might be a photo or an artistic painting, of most likely the "Dangerous turn to the left" sign.

##### Photos made by me

In Hungary we have some traffic signs which are (almost) identical to the German ones.

![Privilaged road][myprivilagedroad]

This is an image made from far away, but it is still bigger than 32x32, so it should be ok.

![Speed limit: 30km/h][my30]

This image is a little dark and skewed, but otherwise fine.

![Traffic signals][mytrafficsignals]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the thirteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			|		Prediction			| 
|:---------------------:|:---------------------------------------------:| 
| End of all speed and passing limits | End of all speed and passing limits |
| No passing | ~~Priority Road~~ |
| Road work | Road work |
| Turn right ahead | Turn right ahead |
| Dangerous curve to the left | ~~Bumpy road~~ |
| Priority road | Priority road |
| Speed limit (30km/h) | Speed limit (30km/h) |
| Traffic signals | Traffic signals |


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This is lower than the test accuracy of 97%, however I would not draw conclusions from this very small (8 element) dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the thirteenth cell of the Ipython notebook.

For the first image, the model is very confident (96%), and correct. The top five soft max probabilities were:

![Top 5 Softmax Probabilities][bars1]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.9561 | End of all speed and passing limits |
| 0.0437 | End of no passing |
| 0.0001 | End of speed limit (80km/h) |
| 0.0001 | Keep left |
| 0.0000 | Children crossing |

For the second image, the model is relatively sure (64%) that we are looking at a Priority Road sign, but the correct answer is No passing which was not even in the top five guesses. Maybe the image was too skewed.

![Top 5 Softmax Probabilities][bars2]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.6373 | Priority road |
| 0.3360 | End of no passing |
| 0.0156 | End of all speed and passing limits |
| 0.0108 | Speed limit (30km/h) |
| 0.0001 | Roundabout mandatory |

For the third image, the model is completely confident (100%), and correct.

![Top 5 Softmax Probabilities][bars3]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000 | Road work |
| 0.0000 | Bicycles crossing |
| 0.0000 | Beware of ice/snow |
| 0.0000 | Double curve |
| 0.0000 | Keep left |

For the fourth image, the model is relatively sure (77%), and correct.

![Top 5 Softmax Probabilities][bars4]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.7743 | Turn right ahead |
| 0.1790 | Roundabout mandatory |
| 0.0226 | Speed limit (60km/h) |
| 0.0067 | Ahead only |
| 0.0042 | Speed limit (30km/h) |


For the fifth image, the model is very sure (99%) that it is a Bumpy road sign, but the correct answer is Dangerous turn to the left, which was not even in the top five guesses.

![Top 5 Softmax Probabilities][bars5]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999 | Bumpy road |
| 0.0000 | Bicycles crossing |
| 0.0000 | Traffic signals |
| 0.0000 | Road work |
| 0.0000 | Slippery road |


For the sixth image, the model is completely confident (100%), and correct.

![Top 5 Softmax Probabilities][bars6]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000 | Priority road |
| 0.0000 | Roundabout mandatory |
| 0.0000 | Keep right |
| 0.0000 | Speed limit (20km/h) |
| 0.0000 | Speed limit (30km/h) |


For the seventh image, the model is completely confident (100%), and correct.

![Top 5 Softmax Probabilities][bars7]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000 | Speed limit (30km/h) |
| 0.0000 | Speed limit (50km/h) |
| 0.0000 | Speed limit (20km/h) |
| 0.0000 | Speed limit (70km/h) |
| 0.0000 | Speed limit (80km/h) |

For the eighth image, the model is completely confident (100%), and correct.

![Top 5 Softmax Probabilities][bars8]

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000 | Traffic signals |
| 0.0000 | Bumpy road |
| 0.0000 | Road work |
| 0.0000 | General caution |
| 0.0000 | Bicycles crossing |

#### Feature map visualization

![End of no passing sign][endofnopassing]
![Feature maps][featuremaps]

In the visualization of the activations of the first convolutional layer, I have seen that different feature maps look for different edges, for example the 7th feature map is activated by diagonal edges, and the 8th feature map is activated by horizontal edges.

#### Reference

[1] Lecun(1998): [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

[2] Sermanet(2011): [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

[3] Ciresan (2012): [Multi-Column Deep Neural Network for Traffic Sign Classification](http://people.idsia.ch/~juergen/nn2012traffic.pdf)


