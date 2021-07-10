import ast
import numpy as np
import pandas as pd
import plotly.express as px
import urllib.request

import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.initializers import *

WIDTH = 224
HEIGHT = 224
LEARNING_RATE = 0.0001
CLASSES = 30
SEED = 120

classes = [
    'banh_mi_nuong', 
    'com_chay', 
    'sup_cua', 
    'bun_dau_mam_tom', 
    'cao_lau', 
    'banh_can', 
    'bot_chien', 
    'nem_chua', 
    'pho', 
    'bun_cha', 
    'bun_bo_hue', 
    'banh_trang_nuong', 
    'banh_mi', 
    'banh_trang_tron', 
    'banh_xeo', 
    'com_tam', 
    'banh_beo', 
    'banh_gio', 
    'banh_khot', 
    'chuoi_chien', 
    'bun_thit_nuong', 
    'banh_cuon', 
    'mi_quang', 
    'banh_bot_loc', 
    'goi_cuon', 
    'chao_long', 
    'xoi_gac', 
    'xoai_lac', 
    'pha_lau', 
    'bap_xao'
]

with open("food.txt", "r", encoding="utf-8") as f:
     food = ast.literal_eval(f.read())

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    return img

def plot_probs(outputs):
    probs = pd.Series(np.round(outputs * 100, 2), classes)
    probs = probs.sort_values(ascending=False).reset_index()
    probs.columns = ['Class', 'Probability']
    fig = px.bar(probs, x='Class', y='Probability')
    fig.update_layout(xaxis_tickangle=-55)
    fig.update_xaxes(title='')
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown(
        "<h1 style='text-align: center;'>Vietnamese Foods Classification</h1> ",
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <center>
            <img 
                src='https://image.thanhnien.vn/1024/uploaded/daly/2018_12_11/2_iwse.jpg' 
                style='width: 95%;'
            >
        </center><br/>
        ''',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose a file")
    url = st.text_input(
    	'Image Url: ', 
    	'https://cdn.tgdd.vn/Files/2020/03/09/1241004/3-mon-banh-mi-kep-la-mieng-hap-dan-thom-ngon-kho-cuong-13.jpg'
    )
    st.write('')
    st.write('')

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.image(bytes_data, use_column_width=True)
        with open('./test.jpg', 'wb') as f: 
            f.write(bytes_data)
    elif url:
        urllib.request.urlretrieve(url, './test.jpg')
        st.markdown(
            f"<center><img src='{url}' style='width: 95%;'></center>",
            unsafe_allow_html=True
        )

    img_test = preprocess_image('./test.jpg')
    
    
    base_model = EfficientNetB0(include_top = False, weights = 'imagenet', input_shape = (WIDTH, HEIGHT, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1280, activation = 'relu', kernel_initializer = glorot_uniform(SEED), bias_initializer = 'zeros')(x)
    x = BatchNormalization()(x)
    predictions = Dense(CLASSES, activation = 'softmax', kernel_initializer = 'random_uniform', bias_initializer = 'zeros')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    for layer in model.layers: layer.trainable = True
    model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model_path = 'model/best_model.h5'

    model.load_weights(model_path)
    pred_probs = model.predict(img_test)[0]

    index = np.argmax(pred_probs)
    label = classes[index]
    
    st.markdown(food[label])
    
    st.markdown(f"**Probability:** {pred_probs[index] * 100:.2f}%")

    plot_probs(pred_probs)

if __name__ == "__main__":
    main()
