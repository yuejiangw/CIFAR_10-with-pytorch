import sys
sys.path.append('..')
import streamlit as st
from predict import Predictor
from model import LeNet
import torch as t

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
st.title("You draw, I guess.")
st.write("All types:")
st.write(classes)

upload_img = st.file_uploader("Please upload your image", type="jpg")

if upload_img is not None:
    # 展示图片
    st.image(upload_img)

    net = LeNet()
    model_path = '../model/state_dict'
    net.load_state_dict(t.load(model_path))
    predictor = Predictor(net, classes)
    result = predictor.predict(upload_img)
    st.write("The result is", classes[result])