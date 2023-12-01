#importing libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

#Title
st.write("""
          # Malaria Cell Classification
          """
          )

#upload file
upload_file = st.sidebar.file_uploader("Upload Cell images", type = "png")

#side button
Generate_pred = st.sidebar.button("Predict")

#importing the model
model = tf.keras.models.load_model("C:/Users/user/Downloads/malaria_det.h5")

#conditions
def import_n_pred(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred

if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['Parasitized', 'Uninfected']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))