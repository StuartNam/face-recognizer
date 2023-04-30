# About this project
In this small project, I built a Face Recognizer on PyTorch and OpenCV.

# Facial Recognition problem
  Facial Recognition is a well-known task in Computer Vision, but it is not a trivial problem. The mission is to identify the identity of the person in the given image. Its result can be used in many application, for example, FaceID for smart phones, security and surveillance for private house or organization, automatically checking attendance for school, etc.\
  Facial recognition is a complicated process, but in an abstract view, it consists of two main steps:
  - Facial detection: Given an image, the process locates the faces with in that image.
  - Facial recognition: Given the face image, the process identify the person in that image.
  Each of the steps require different approach and implementation.
# Main components of a face recognizer
## 1. Face detector
Newly introduced in 2021, YuNet has been proven to be very lightweight but still able to give significant result compared to the other models in the field of Facial Detection. It is available as an open-source model.

## 2. Face recognizer
The facial recognizing step required 2 components:
### 2.1. Feature extractor
This feature extractor uses convolutional-based architecture (which is specialized on image feature extracting). It receives the face image as input and gives the output as high-level and abstract feature in the form of Feature vectors.
### 2.2. Feature classifier
Final step on Facial recognizing, the Feature classifier classify the identity of the given image based on the Feature vectors. It can be modeled as a fully-connected Neural Network with Softmax layer at the output for identify the given face image from the Feature vectors.

# Data collection
For training the model, I collect the data through the camera of the PC device. Each of the collected data will be pre-processed through a fixed pipeline and store in the data folder for future training.\
\

Having fun!
Nam Ha
