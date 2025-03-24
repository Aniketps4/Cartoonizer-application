import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Cartoonizer App")

# Function to convert image to cartoon
def cartoonize(img, line_size=5, blur_value=7, k=9):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_size, blur_value)
    
    # Reduce color palette
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()].reshape(img.shape)
    
    # Apply bilateral filter
    blurred = cv2.bilateralFilter(result, d=3, sigmaColor=200, sigmaSpace=200)
    
    # Combine edge mask with color-quantized image
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    cartoon_image = cartoonize(image)
    
    st.image(cartoon_image, caption="Cartoonized Image", use_column_width=True)
