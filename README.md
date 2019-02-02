# Emotion Recognition


Emotion recognition using Convolution Neural Networks


## Motivation

Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. Our facial emotions are expressed through activation of specific sets of facial muscles. These sometimes subtle, yet complex, signals in an expression often contain an abundant amount of information about our state of mind. Through facial emotion recognition, we are able to measure the effects that content and services have on the audience/users through an easy and low-cost procedure. For example, retailers may use these metrics to evaluate customer interest. Healthcare providers can provide better service by using additional information about patients' emotional state during treatment. Entertainment producers can monitor audience engagement in events to consistently create desired content.


## The Database


The dataset I used for training the model is from a Kaggle Facial Expression Recognition Challenge a few years back (FER2013). It comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each labeled with one of the 7 emotion classes:anger, disgust, fear, happiness, sadness, surprise, and neutral.

As I was exploring the dataset, I discovered an imbalance of the “disgust” class (only 113 samples) compared to many samples of other classes. I decided to merge disgust into anger given that they both represent similar sentiment. To prevent data leakage, I built a data generator [fer2013datagen.py](https://github.com/Siddhantsaka/Emotion_recognition/blob/master/fer2013datagen.py) that can easily separate training and hold-out set to different files. I used 28709 labeled faces as the training set and held out the remaining two test sets (3589/set) for after-training validation. The resulting is a 6-class, balanced dataset, shown in Figure 2, that contains angry, fear, happy, sad, surprise, and neutral. Now we’re ready to train.

## The Model
Deep learning is a popular technique used in computer vision. I chose convolutional neural network (CNN) layers as building blocks to create my model architecture. CNNs are known to imitate how the human brain works when analyzing visuals.
A typical architecture of a convolutional neural network will contain an input layer, some convolutional layers, some dense layers (aka. fully-connected layers), and an output layer. These are linearly stacked layers ordered in sequence. In [Keras](https://keras.io/models/sequential/), the model is created as `Sequential()` and more layers are added to build architecture.

### The input layer
+ The input layer has pre-determined, fixed dimensions, so the image must be __pre-processed__ before it can be fed into the layer. I used [OpenCV](http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0), a computer vision library, for face detection in the image. The `haar-cascade_frontalface_default.xml` in OpenCV contains pre-trained filters and uses `Adaboost` to quickly find and crop the face.
+ The cropped face is then converted into grayscale using `cv2.cvtColor` and resized to 48-by-48 pixels with `cv2.resize`. This step greatly reduces the dimensions compared to the original RGB format with three color dimensions (3, 48, 48).  The pipeline ensures every image can be fed into the input layer as a (1, 48, 48) numpy array.
### Convolutional Layers
+ The numpy array gets passed into the `Convolution2D` layer where I specify the number of filters as one of the hyperparameters. The __set of filters__(aka. kernel) are unique with randomly generated weights. Each filter, (3, 3) receptive field, slides across the original image with __shared weights__ to create a __feature map__.
+  __Convolution__ generates feature maps that represent how pixel values are enhanced, for example, edge and pattern detection.

+ __Pooling__ is a dimension reduction technique usually applied after one or several convolutional layers. It is an important step when building CNNs as adding more convolutional layers can greatly affect computational time. I used a popular pooling method called `MaxPooling2D` that uses (2, 2) windows across the feature map only keeping the maximum pixel value. The pooled pixels form an image 
with dimentions reduced by 4.

### Dense Layers
+ The dense layer, is inspired by the way neurons transmit signals through the brain. It takes a large number of input features and transform features through layers connected with trainable weights.

+ These weights are trained by forward propagation of training data then backward propagation of its errors. __Back propagation__ starts from evaluating the difference between prediction and true value, and back calculates the weight adjustment needed to every layer before. We can control the training speed and the complexity of the architecture by tuning the hyper-parameters, such as __learning rate__ and __network density__. As we feed in more data, the network is able to gradually make adjustments until errors are minimized. 
+ Essentially, the more layers/nodes we add to the network the better it can pick up signals. As good as it may sound, the model also becomes increasingly prone to overfitting the training data. One method to prevent overfitting and generalize on unseen data is to apply __dropout__. Dropout randomly selects a portion (usually less than 50%) of nodes to set their weights to zero during training. This method can effectively control the model's sensitivity to noise during training while maintaining the necessary complexity of the architecture.



### Output Layer
+ Instead of using sigmoid activation function, I used **softmax** at the output layer. This output presents itself as a probability for each emotion class.
+ Therefore, the model is able to show the detail probability composition of the emotions in the face. As later on, you will see that it is not efficient to classify human facial expression as only a single emotion. Our expressions are usually much complex and contain a mix of emotions that could be used to accurately describe a particular expression.

## Model Validation
### Performance
As it turns out, the final CNN had a __validation accuracy of 69%__. This actually makes a lot of sense. Because our expressions usually consist a combination of emotions, and _only_ using one label to represent an expression can be hard. In this case, when the model predicts incorrectly, the correct label is often the __second most likely emotion__.



## References

1. [*"Dataset: Facial Emotion Recognition (FER2013)"*](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

2. [*"Andrej Karpathy's Convolutional Neural Networks (CNNs / ConvNets)"*](http://cs231n.github.io/convolutional-networks/) Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.

3. Srivastava et al., 2014. *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"*, Journal of Machine Learning Research, 15:1929-1958.

4. Duncan, D., Shine, G., English, C., 2016. [*"Report: Facial Emotion Recognition in Real-time"*](http://cs231n.stanford.edu/reports2016/022_Report.pdf) Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.



