import sys
sys.path.append('..')
import streamlit as st
from predict import Predictor
from model import LeNet, Vgg16_Net
import torch as t

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
st.title("You draw, I guess.")
st.write("All types:")
st.write(classes)

upload_img = st.file_uploader("Please upload your image", type="jpg")

if upload_img is not None:
    # 展示图片
    st.image(upload_img)

    net = Vgg16_Net()
    model_path = '../model/state_dict'
    net.load_state_dict(t.load(model_path, map_location=t.device('cpu')))
    predictor = Predictor(net, classes)
    result = predictor.predict(upload_img)
    st.write("The result is", classes[result])