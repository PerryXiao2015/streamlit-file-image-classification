#https://github.com/streamlit/streamlit/issues/511
#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st
import cv2 
import numpy as np
import pandas as pd
from PIL import Image,ImageEnhance

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import time


# import the models for further classification experiments
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

import matplotlib.pyplot as plt

# imports for reproducibility
import tensorflow as tf
import random
import os

def vgg16_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    # print predictions
    # convert the probabilities to class labels
    # we will get top 5 predictions which is the default
    label_vgg = decode_predictions(predictions)
    # print VGG16 predictions
    #for prediction_id in range(len(label_vgg[0])):
    #    print(label_vgg[0][prediction_id])
    
    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame

def resnet50_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # prepare the image for the ResNet50 model
    processed_image = resnet50.preprocess_input(image_batch.copy())
    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    # convert the probabilities to class labels
    # If you want to see the top 3 predictions, specify it using the top argument
    label_resnet = decode_predictions(predictions, top=3)
    # print ResNet predictions
    #for prediction_id in range(len(label_resnet[0])):
    #    print(label_resnet[0][prediction_id])
    
    # format final image visualization to display the results of experiments
    #cv2.putText(cam_frame, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]) , (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    #cv2.putText(cam_frame, "MobileNet: {}, {:.2f}".format(label_mobilenet[0][0][1], label_mobilenet[0][0][2]) , (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    #cv2.putText(cam_frame, "Inception: {}, {:.2f}".format(label_inception[0][0][1], label_inception[0][0][2]) , (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(cam_frame, "ResNet50: {}, {:.2f}".format(label_resnet[0][0][1], label_resnet[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    

def mobilenet_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # prepare the image for the MobileNet model
    processed_image = mobilenet.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)

    # convert the probabilities to imagenet class labels
    label_mobilenet = decode_predictions(predictions)
    # print MobileNet predictions
    #for prediction_id in range(len(label_mobilenet[0])):
    #    print(label_mobilenet[0][prediction_id])
    
    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "MobileNet: {}, {:.2f}".format(label_mobilenet[0][0][1], label_mobilenet[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    
    
def inception_v3_predict(cam_frame, image_size):
    frame= cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = inception_v3.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    # print predictions
    # convert the probabilities to class labels
    # we will get top 5 predictions which is the default
    label_inception = decode_predictions(predictions)
    # print Inception predictions
    #for prediction_id in range(len(label_inception[0])):
    #    print(label_inception[0][prediction_id])
    
    # format final image visualization to display the results of experiments
    cv2.putText(cam_frame, "Inception: {}, {:.2f}".format(label_inception[0][0][1], label_inception[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return cam_frame    

mode = 1
model = vgg16.VGG16(weights='imagenet')
image_size = 224

def select_models():
    st.sidebar.markdown("# Image Classification")
    option = st.sidebar.selectbox(
         'Select a Deep Learning Model:',
         ["VGG16","RESNET50","MOBILENET","INCEPTION_V3"], index=0)
    st.sidebar.write('You selected:', option)
    if option == "VGG16":
        model = vgg16.VGG16(weights='imagenet')
        image_size = 224
        mode = 1
    elif option == "RESNET50":
        model = resnet50.ResNet50(weights='imagenet')
        image_size = 224
        mode = 2
    elif option == "MOBILENET":
        model = mobilenet.MobileNet(weights='imagenet')
        image_size = 224
        mode = 3
        
    elif option == "INCEPTION_V3":
        model = inception_v3.InceptionV3(weights='imagenet')
        image_size = 299
        mode = 4
    return mode

def classify_image(frame,mode):
    if mode == 1:
        frame = vgg16_predict(frame, image_size)
    elif mode == 2:
        frame = resnet50_predict(frame, image_size)
    elif mode == 3:
        frame = mobilenet_predict(frame, image_size)
    elif mode == 4:
        frame = inception_v3_predict(frame, image_size)
    return frame

def main():
    """Image Classification App"""

    st.title("Image Classification App")
    st.text("Build with Streamlit and OpenCV")
    activities = ["Image Classification","About"]
    choice = st.sidebar.selectbox("Select Activty",activities)
    if choice == 'Image Classification':
        mode = select_models()
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)
            #convert to CV2 format
            new_img = np.array(our_image.convert('RGB'))
            image = cv2.cvtColor(new_img,1)
            # Get the boxes for the objects detected by YOLO by running the YOLO model.
            image = classify_image(image,mode)
            st.text("Classification Image")
            st.image(image.astype(np.uint8), use_column_width=True)

    elif choice == 'About':
        st.subheader("About Image Classification App")
        st.markdown("Built with Streamlit by [LSBU](https://www.lsbu.ac.uk/)")
        st.text("Professor Perry Xiao")
        st.success("Copyright @ 2020 London South Bank University")
if __name__ == '__main__':
    main()	
