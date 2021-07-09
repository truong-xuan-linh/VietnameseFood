import ast
import numpy as np
import pandas as pd
import plotly.express as px
import urllib.request

import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.models import*
from tensorflow.keras.preprocessing import image


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

with open("food.txt") as f:
    info = ast.literal_eval(f.read())


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


st.markdown(
    "<h1 style='text-align: center;'>Vietnamese Foods Classification</h1> ",
    unsafe_allow_html=True
)

st.markdown(
    '''
    <center>
        <img 
            src='https://www.google.com/logos/doodles/2020/celebrating-banh-mi-6753651837108330.3-2xa.gif' 
            style='width: 90%;'
        >
    </center><br/>
    ''',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose a file")
url = st.text_input(
	'Image Url: ', 
	'https://upload.wikimedia.org/wikipedia/commons/5/53/Pho-Beef-Noodles-2008.jpg'
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
        f"<center><img src='{url}' style='width: 90%;'></center>",
        unsafe_allow_html=True
    )

img_test = preprocess_image('./test.jpg')
model = create_model()
model.load_weights('final_model.h5')
pred_probs = model.predict(img_test)[0]
print(pred_probs)

index = np.argmax(pred_probs)
label = classes[index]

st.markdown(food[label])

plot_probs(pred_probs)
