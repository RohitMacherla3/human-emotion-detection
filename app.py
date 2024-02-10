import streamlit as st
import onnxruntime as rt
import cv2
import numpy as np
from PIL import Image, ImageOps
import time

st.set_page_config(layout="centered")

def load_Model():
    providers = ['CPUExecutionProvider']
    ViT_model = rt.InferenceSession('Model/ViT_quantized.onnx', providers=providers)
    return ViT_model

Model = load_Model()

st.markdown(
    """
    <h1 style='text-align: center;'>Human Emotion Detection</h1>
    """,
    unsafe_allow_html=True
)

inp_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def Emotion_Detector(image_array, model):
    im = cv2.resize(image_array, (224, 224))
    if im.shape[-1] > 3:
        im = im[..., :3]
    if im.shape[-1] == 1 or len(im.shape) ==2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = np.float32(im)
    im = np.expand_dims(im, axis=0)
    
    time_start = time.time()

    prediction = model.run(['dense'], {'input': np.array(im)})
    emotion_class = np.argmax(prediction, axis =-1)[0][0]
    
    time_elapsed = time.time() - time_start
    
    if int(emotion_class) == 0:
        emotion = 'Angry'
    elif int(emotion_class) == 1:
        emotion = 'Happy'
    elif int(emotion_class) == 2:
        emotion = 'Sad'
    
    return {'Emotion':  emotion, 
            'TimeElapsed': str(time_elapsed)}

def predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    img = np.array(image)
    
    return Emotion_Detector(img, model)

if inp_image is not None:
    
    st.image(inp_image, use_column_width=600)
    
    st.markdown(
        """
        <style>
        /* Increase the size of the button */
        .stButton>button {
            width: 200px; 
            height: 50px; 
            margin: 0 auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            image = Image.open(inp_image)
            predictions = predict(image, Model)
            st.success(predictions)