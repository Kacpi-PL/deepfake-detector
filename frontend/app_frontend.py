import streamlit as st
import requests
from PIL import Image


st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("StyleGAN3 Face detector")
st.write("Project: image classification system based on Resnet50 and pytorch")
st.markdown("---")

# Defining API Url
FASTAPI_URL = "http://backend:8000/predict/"

# defining image upload widget to load image to memory
uploaded_file = st.file_uploader(
    label="Upload image containing face here", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width='content')

    if st.button("Start detector", type="primary"):
        with st.spinner("Analyzing"):
            try:
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                file_type = uploaded_file.type

                files = {"file": (file_name, file_bytes, file_type)}

                # Sending post to FASTApi
                response = requests.post(FASTAPI_URL, files=files)

                # right response
                if response.status_code == 200:
                    result = response.json()

                    prediction = result.get("prediction")
                    confidence = result.get("confidence")
                    filename = result.get("filename")

                    st.subheader("Classification Result:")

                    # Changing site look depending on pred.
                    if "Fake" in prediction:
                        st.error(
                            f"**StyleGAN3 deepfake detected** (Klasa: {prediction})"
                        )
                    else:
                        st.success(
                            f"**No StyleGAN3 deepfake detected** (Klasa: {prediction})"
                        )

                    # Confidence score
                    st.metric(label="Confidence Score", value=confidence)
                    st.caption(f"For file: {filename}")

                else:
                    st.error(f"Fastapi error: code {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI! ")
