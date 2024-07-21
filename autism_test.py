import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import cv2
import base64


st.set_page_config(
    page_title="Autism Prediction",
    page_icon=":stethoscope:",
)


# Add background
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Background image file '{image_file}' not found.")


if __name__ == '__main__':
    add_bg_from_local('ak.png')

# Load the trained model
model = tf.keras.models.load_model('autism_1.h5')


# Define a function to preprocess the image
def preprocess_image(img):
    img_resized = resize(img, (150, 150), anti_aliasing=True)  # Resize image to 150x150
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:  # If image has 3 channels, convert to grayscale
        img_resized = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if len(img_resized.shape) == 2:  # If image is grayscale, add a channel dimension
        img_resized = np.expand_dims(img_resized, axis=-1)
    img_reshape = img_resized.reshape(1, 150, 150, 1)  # Add batch dimension
    return img_reshape


# Capture a frame from webcam
def capture_frame():
    webcam = cv2.VideoCapture(2)
    if not webcam.isOpened():
        st.error("Webcam not accessible")
        return None
    ret, frame = webcam.read()
    webcam.release()
    if not ret:
        st.error("Failed to capture image")
        return None
    return frame


# Initialize session state
if 'captured_img' not in st.session_state:
    st.session_state['captured_img'] = None

# Streamlit app interface
st.title(' :blue[Autism Detection from Facial Image]')

tab1, tab2, tab3, tab4 = st.tabs(
    [":page_with_curl: **HOMEPAGE**", ":reminder_ribbon: ABOUT AUTISM", ":medical_symbol: **PREDICTION**",
     ":scroll: **CONCLUSION**"])

with tab1:
    st.markdown("""
    ## Welcome to the Autism Detection App

    This application leverages the power of deep learning to aid in the early detection of Autism Spectrum Disorder (ASD) through facial image analysis. By uploading a photo or capturing an image via your webcam, you can get a prediction about whether a child might have ASD. Early detection is crucial for providing timely interventions and support.

    ### Key Features
    - **Upload Image**: Upload a facial image in JPEG, JPG or PNG format for analysis.
    - **Capture Image**: Use your webcam to capture an image and analyze it instantly.
    - **Deep Learning Model**: The app uses a Convolutional Neural Network (CNN) trained on a dataset of facial images to make predictions.

    ### How to Use
    1. Navigate to the **Prediction** tab.
    2. Choose between **Upload Image** or **Capture Image**.
    3. Follow the instructions to upload or capture an image.
    4. View the prediction result on the same page.

    ### Disclaimer
    This tool is designed for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your healthcare provider with any questions you may have regarding a medical condition.

    ### About the Project
    This project is part of ongoing research to explore the potential of deep learning in healthcare. I hope to contribute to the field of autism detection and support further research and development.
    """)

    st.write("Created by :blue[ADEESH KRISHNA] :registered:")

with tab2:
    st.header("Autism Spectrum Disorder")
    st.write("""
    :blue[**Autism Spectrum Disorder (ASD)**]  is a complex developmental condition that affects communication, behavior, and social interactions. It is known as a "Spectrum" disorder because it encompasses a wide range of symptoms and abilities. Individuals with autism may experience challenges in different ways, and the severity of symptoms can vary significantly.
    """)

    st.subheader("Autism Spectrum Disorder (ASD)")
    st.video("https://youtu.be/6jUv3gDAM1E?si=X9Yq0T4Xlbri4SNc", format="video/mp4")

    st.markdown("""
    ### Understanding Autism Spectrum Disorder (ASD)

    ### Key Characteristics of Autism
    - **Communication Difficulties**: Individuals with autism may have trouble with verbal and non-verbal communication. This can include delayed speech development, difficulty in understanding language, and challenges with expressing thoughts and emotions.
    - **Social Interaction Challenges**: People with autism often find it difficult to engage in typical social interactions. They may struggle with understanding social cues, making eye contact, and forming relationships.
    - **Repetitive Behaviors**: Repetitive behaviors and restricted interests are common in autism. This can include repetitive movements, strict adherence to routines, and intense focus on specific topics or activities.
    - **Sensory Sensitivities**: Many individuals with autism are sensitive to sensory input such as light, sound, texture, and temperature. They may overreact or underreact to sensory stimuli.

    ### Importance of Early Detection
    Early detection of autism is crucial for providing timely interventions that can significantly improve outcomes. The earlier autism is identified, the sooner appropriate support and therapies can be implemented, helping individuals develop essential skills and reach their full potential.

    ### Benefits of Early Intervention
    - **Improved Communication Skills**: Early intervention can help children develop better communication skills, both verbal and non-verbal.
    - **Enhanced Social Abilities**: With early support, children can learn social skills that improve their ability to interact with others and form meaningful relationships.
    - **Behavioral Support**: Early intervention programs can address behavioral challenges and help children develop positive behaviors and coping strategies.
    - **Family Support**: Early detection and intervention provide families with the resources and knowledge needed to support their child's development effectively.

    :blue[ **Accept  Understand   Love**]
    """)

with tab3:
    st.header("Prediction")
    st.write('Upload a facial image or capture a facial image to predict whether a child is autistic or not.')
    tab_1, tab_2 = st.tabs([":large_blue_diamond: Upload Image", ":large_blue_diamond: Capture Image"])

    with tab_1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                img = imread(uploaded_file)  # Read image
                st.image(img, caption='Uploaded Image.', use_column_width=True)

                with st.spinner('Processing...'):
                    # Preprocess the image
                    image = preprocess_image(img)

                if st.button('PREDICT', key='upload_predict'):
                    # Display the result
                    predictions = model.predict(image)
                    prediction = np.argmax(predictions, axis=1)  # Get the predicted class
                    if prediction == 1:
                        st.success("The model predicts: **Child is not Autistic**")
                    else:
                        st.success("The model predicts: **Child is Autistic**")


            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload an image file.")

    with tab_2:
        st.write("**Prepare for Image Capture**")
        st.write("Please ensure you are in a well-lit area and when you are ready, click the button below to start capturing.")
        if st.button("CAPTURE", key='capture'):
            image = capture_frame()
            if image is not None:
                st.session_state['captured_img'] = image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # st.image(image, caption='Captured Image.', use_column_width=True)

        if st.session_state['captured_img'] is not None:
            try:
                image = cv2.cvtColor(st.session_state['captured_img'], cv2.COLOR_BGR2RGB)
                st.image(image, caption='Captured Image.', use_column_width=True)

                with st.spinner('Processing...'):
                    # Preprocess the image
                    img_resized = resize(image, (150, 150, 1),
                                         anti_aliasing=True)  # Resize image to match model input shape
                    img_reshape = img_resized.reshape(1, 150, 150, 1)

                if st.button('PREDICT', key='capture_predict'):
                    # Display the result
                    predictions = model.predict(img_reshape)
                    prediction = np.argmax(predictions, axis=1)  # Get the predicted class
                    if prediction == 1:
                        st.success("The model predicts: **Child is not Autistic**")
                    else:
                        st.success("The model predicts: **Child is Autistic**")

            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab4:
    st.header("Conclusion")
    st.write("""
    This project demonstrates the potential of using deep learning for the early detection of Autism Spectrum Disorder (ASD) based on facial images.
    While the model shows promising results, it is important to note that it is not a diagnostic tool. Further validation and testing
    with larger and more diverse datasets are necessary to improve the model's accuracy and reliability.

    Future work could involve integrating this model into a comprehensive screening tool that combines various data sources,
    including behavioral assessments and medical history, to provide a more holistic view of the child's development.

    We hope this project sparks further research and development in the field of autism detection and contributes to the overall efforts
    in improving the quality of life for individuals with ASD.
    """)

    st.link_button(':link: Link to my Colab Notebook',
                   "https://colab.research.google.com/drive/1vyf54LvRAlDJPFfBrrz5kl7__IFYOS9u", help='Colab Notebook',
                   type="secondary", disabled=False, use_container_width=False)
    st.link_button(':link: Link to Research Paper',
                   "https://www.researchgate.net/publication/350396741_Detecting_autism_from_facial_image",
                   help='Research Paper', type="secondary", disabled=False, use_container_width=False)
