from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np


import streamlit as st
from PIL import Image
from skimage.transform import resize


# Path del modelo preentrenado
MODEL_PATH = 'models/model_RS50_Aves.h5'
    
width_shape = 224
height_shape = 224

names = ['AFRICAN FIREFINCH','ALBATROSS','ALEXANDRINE PARAKEET','AMERICAN AVOCET','AMERICAN BITTERN',
         'AMERICAN COOT','AMERICAN GOLDFINCH','AMERICAN KESTREL','AMERICAN PIPIT','AMERICAN REDSTART']



def model_prediction(img, model):

    img_resize = resize(img, (width_shape, height_shape))
    x=preprocess_input(img_resize*255)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds


def main():
    
    model=''


    if model=='':
        model = load_model(MODEL_PATH)
    
    st.title("Clasificador de Aves :sunglasses:")

    predictS=""
    img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
    
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))    
        st.image(image, caption="Imagen", use_column_width=False)
    
    if st.button("Predicción"):
         predictS = model_prediction(image, model)
         st.success('LA CLASE ES: {}'.format(names[np.argmax(predictS)]))


if __name__ == '__main__':
    main()

