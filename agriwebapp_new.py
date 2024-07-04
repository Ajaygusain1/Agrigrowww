

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:47:43 2023

@author: DELL
"""

import seaborn as sns
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

import tensorflow as tf
import os
import json
import numpy as np


# loading the saved models


from PIL import Image



pk1 = open("D:\saved models-20230525T165545Z-001\saved models\RandomForest.pkl",'rb')
RF=pickle.load(pk1)

pk2=open("D:\\saved models-20230525T165545Z-001\\saved models\\NBClassifier_fert2.pkl",'rb')
RF2=pickle.load(pk2)


#working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"D:\saved models-20230525T165545Z-001\saved models\plant_disease_prediction_model (1).h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"D:\saved models-20230525T165545Z-001\saved models\class_indices.json"))



# sidebar for navigation
with st.sidebar:
    
    image = Image.open("D:\saved models-20230525T165545Z-001\saved models\wheat.jpg")

    st.image(image)
    
    selected = option_menu('AGRIGROW',
                          
                          ['Home',
                           'Crop Prediction',
                           'Fertilizer Prediction',
                           'Disease Classifier'
                           
                           ],
                          #icons=['activity','biohazard_sign','person'],
                          default_index=0)
    



def pred(N,P,K, temp,humid,ph,rain):
    pred=RF.predict([[N,P,K, temp,humid,ph,rain]])
    print(pred)
    return pred


def pred2(temparature,humidity,moisture,Soil_Type,Crop_Type,nitrogen,potassium,phosphorous):
    pred2=RF2.predict([[temparature,humidity,moisture,Soil_Type,Crop_Type,nitrogen,potassium,phosphorous]])
    print(pred2)
    return pred2


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name



def main():

    if (selected == 'Home'):
     #   image1=Image.open('D:/saved models-20230525T165545Z-001/saved models/Pantnagar_logo.jpg')
      #  left_co, cent_co,last_co = st.columns(3)
       # with cent_co:
        #    st.image(image1)
    
      # image1=Image.open('D:/saved models-20230525T165545Z-001/saved models/Pantnagar_logo.jpg')
        original_title = '<p style="font-family:Serif; color:Black; text-align: center; font-size: 50px;">Invertis University, Bareilly</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        original_title1 = '<p style="font-family:Serif; color:Black; text-align: center; font-size: 36px;">Department Of Computer Engineering</p>'
        st.markdown(original_title1, unsafe_allow_html=True)
        original_title2 = '<p style="font-family:Serif; color:Black; text-align: center; font-size: 25px;">Project Guide : Miss Purnima Awasthi</p>'
        st.markdown(original_title2, unsafe_allow_html=True)
        
       

    if (selected == 'Crop Prediction'):
        
        # page title
        st.title('Crop Prediction using ML')
        
        
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)
        
        with col1:
            N = st.text_input('Nitrogen')
            
        with col2:
            P = st.text_input('Phosphorus')
        
        with col3:
            K = st.text_input('Potassium')
        
        with col1:
            temp = st.text_input('Temperature')
        
        with col2:
           humid = st.text_input('Humidity')
        
        with col3:
            ph = st.text_input('ph value')
        
        with col1:
            rain = st.text_input('Rainfall')
        
     
        
        # code for Prediction
        result = ''
        
        if st.button("predict"):
            result=pred(N,P,K, temp,humid,ph,rain)
        st.success("crop predicted is : {}".format(result))
        
        image = Image.open("D:\saved models-20230525T165545Z-001\saved models\wheat.jpg")

        st.image(image)
        
    if (selected == 'Fertilizer Prediction'):
         
         # page title
         st.title('Fertilizer Prediction using ML')
         
         # getting the input data from the user
         # Temparature, Humidity , Moisture, Soil_Type, Crop_Type,Nitrogen, Potassium, Phosphorous
         col1, col2, col3 = st.columns(3)
         
         with col1:
             temparature=st.text_input("Temperature")
         with col2:
            humidity=st.text_input("Humidity")
         with col3:
             moisture=st.text_input("Moisture")
         with col1:
                mapping0={'black':0,'clayey':1,'loamy':2,'red':3,'sandy':4}
                Soil_Type= st.selectbox("select Soil ",list(mapping0.keys()))
                Soil_Type=mapping0[Soil_Type]
         with col2:
             mapping1={'Sugarcane':8,'Cotton':1,'Millets':4,'Paddy':6,'Pulses':7,'Wheat':10,'Tobacco':9,'Barley':0,'Oil seeds':5,'Ground Nuts':2,'Maize':3}
             Crop_Type= st.selectbox("select Crop ",list(mapping1.keys()))
             Crop_Type=mapping1[Crop_Type]
         with col3:
            nitrogen=st.text_input("Nitrogen")
         with col1:
            potassium=st.text_input("Potassium")
         with col2:
            phosphorous=st.text_input("Phosphorous")
            
         result = ''
        
         if st.button("predict"):
            result=pred2(temparature,humidity,moisture,Soil_Type,Crop_Type,nitrogen,potassium,phosphorous)
         st.success("fertilizer predicted is : {}".format(result))
         
         
    if( selected=="Disease Classifier"):
        st.title('Plant Disease Classifier')

        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((150, 150))
                st.image(resized_img)

            with col2:
                if st.button('Classify'):
                    # Preprocess the uploaded image and predict the class
                    prediction = predict_image_class(model, uploaded_image, class_indices)
                    st.success(f'Prediction: {str(prediction)}')    
        
        
if __name__=='__main__':
        main()