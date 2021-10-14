import sys

from streamlit.proto.RootContainer_pb2 import SIDEBAR
from torch._C import device
sys.path.append('..')
import streamlit as st
from predict import Predictor
from model import LeNet, Vgg16_Net
import torch as t

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
st.title(":sunglasses: CIFAR-10 Playground :+1:")
st.header("All types:")
st.subheader("Plane, Car, Bird, Cat, Deer, Dog, Frog, Hourse, Ship, Truck")


side_bar = st.sidebar
side_bar.title("Select your model")

# select your model
model_select = side_bar.radio(
    'Model Name',
    ('LeNet-5','VGG-16')
)

# select your testing device
device_select = side_bar.radio(
    'Device',
    ('CPU','GPU')
)

upload_img = st.file_uploader("Please upload your image", type="jpg")

if upload_img is not None:
    # 展示图片
    st.image(upload_img)

    # 选择模型
    if model_select == 'VGG-16':
        net = Vgg16_Net()
        model_path = '../model/state_dict_vgg'

    else:
        net = LeNet()
        model_path = '../model/state_dict_le'

    try:
        net.load_state_dict(t.load(model_path, map_location=t.device(device_select.lower())))
        predictor = Predictor(net, classes)
        result = predictor.predict(upload_img)
        st.write("The result is", classes[result])
    except RuntimeError:
        st.write("Please check your device, do not select GPU button when you have no CUDA!")