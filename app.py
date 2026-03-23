import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo Editor", layout="wide")

st.title("📸 Photo Editor using OpenCV & Streamlit")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    # Sidebar Controls
    st.sidebar.header("Resize")
    width = st.sidebar.slider("Width", 100, 1000, img.shape[1])
    height = st.sidebar.slider("Height", 100, 1000, img.shape[0])
    img = cv2.resize(img, (width, height))

    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)

    def adjust_bc(image, brightness, contrast):
        beta = brightness
        alpha = 1 + (contrast / 100)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    img = adjust_bc(img, brightness, contrast)

    st.sidebar.header("Filters")

    # Use checkboxes instead of buttons
    if st.sidebar.checkbox("Grayscale"):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if st.sidebar.checkbox("Blur"):
        img = cv2.GaussianBlur(img, (15, 15), 0)

    if st.sidebar.checkbox("Sharpen"):
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

    if st.sidebar.checkbox("Warm Filter"):
        increase = np.array([10, 0, 0], dtype=np.uint8)
        img = cv2.add(img, increase)

    if st.sidebar.checkbox("Edge Detection"):
        edges = cv2.Canny(img, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    if st.sidebar.checkbox("Cartoon Effect"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img, 9, 250, 250)
        img = cv2.bitwise_and(color, color, mask=edges)

    if st.sidebar.checkbox("Portrait Blur"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = cv2.GaussianBlur(gray, (21,21), 0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        blurred = cv2.GaussianBlur(img, (25,25), 0)
        img = np.where(mask > 127, img, blurred)

    st.subheader("Edited Image")
    st.image(img, use_container_width=True)

    # Download without saving file
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")

    st.download_button(
        label="📥 Download Image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )