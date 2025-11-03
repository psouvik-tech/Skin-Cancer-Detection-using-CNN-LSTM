import streamlit as st
import cv2 as cv
import numpy as np
import keras
import pickle
import os


#label_name = localization_list = ['scalp', 'ear', 'face', 'back', 'trunk', 'chest',
#                      'upper extremity', 'abdomen', 'unknown', 'lower extremity',
#                      'genital', 'neck', 'hand', 'foot', 'acral']
#elif a == " PAD-UFES-20":
#label_name = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC', 'MEL']
#elif a == "ISIC":
label_name = ['actinic keratosis', 'basal cell carcinoma',
                       'dermatofibroma', 'melanoma', 'nevus',
                       'pigmented benign keratosis', 'seborrheic keratosis',
                       'squamous cell carcinoma', 'vascular lesion']
st.write("""This Melanoma Skin Cancer Detection is applied deep learning \
         techniques and transfer learning to leverage the pre-trained \
             knowledge of a base model. The model is trained for different \
                 Skin Cancers based on ISIC, PAD-UFES-20, and HAM10000 \
                     datasets.""")              

#st.write("Please input only leaf Images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato. Otherwise, the model will not work perfectly.")
#print(os.getcwd())
file_path = os.getcwd() + r'\Melanoma_Datasets_CNN_LSTM_model1.pkl'
#file_path = os.getcwd() + r'\Melanoma_Datasets_CNN_LSTM_model2.pkl'
#file_path = os.getcwd() + r'\Melanoma_Datasets_CNN_LSTM_model4.pkl'

#file_path = r'C:\Users\prama\OneDrive\Documents\VT_DellInsprion153000\Spring 2025\Independent Study\Meeting 5-Spring 2025\Melanoma_Datasets_CNN_LSTM_model2.pkl'
#file_path = r'C:\Users\prama\OneDrive\Documents\VT_DellInsprion153000\Spring 2025\Independent Study\Meeting 5-Spring 2025\Melanoma_Datasets_CNN_LSTM_model4.pkl'
#model = keras.models.load_model(file_path)
#model = keras.models.load_model(r'C:\Users\prama\OneDrive\Documents\VT_DellInsprion153000\Spring 2025\Independent Study\Meeting 5-Spring 2025\Melanoma_Datasets_CNN_LSTM_model2.pkl')
#model = keras.models.load_model(r'C:\Users\prama\OneDrive\Documents\VT_DellInsprion153000\Spring 2025\Independent Study\Meeting 5-Spring 2025\Melanoma_Datasets_CNN_LSTM_model4.pkl')

# Load your model from a Pickle file
with open(file_path, 'rb') as file:
    model = pickle.load(file)

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (120, 120)), axis=0)

    # Make predictions using the Pickle-loaded model
    predictions = model.predict(normalized_image)
    st.image(image_bytes)
    if predictions[0][np.argmax(predictions)] * 100 >= 80:
        st.write(f"Result is: {label_name[np.argmax(predictions)]}")
    else:
        st.write("Try Another Image")