import streamlit as st
import requests
from PIL import Image

# Set the FastAPI backend URL (replace with your actual Vercel URL)
backend_url = "http://localhost:5000/predict"  # Change to Vercel URL after deployment

st.set_page_config(page_title="Pose Detection", layout="wide")
# Streamlit App
st.markdown('<div class="center-header">Pose Detection</div>', unsafe_allow_html=True)


# Custom CSS for styling
st.markdown("""
<style>
    .block-container{
        padding-top:1rem;
        padding-bottom:0rem;
        padding-left:1rem;
        padding-right:1rem;
    }
    .center-header {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        margin-bottom: -10px;
        margin-top: 10px;
        font-size: 45px;
        font-weight: bold;
    }
</style>
""",unsafe_allow_html=True)
# Create two columns
col1, col2 = st.columns([0.3,0.7])


# Upload image
with col1:
    uploaded_files = st.file_uploader("Upload an image for prediction", type=["jpg", "png"], accept_multiple_files=True)

# if uploaded_file is not None:
    if uploaded_files:
        # Display uploaded image
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        # for uploaded_file in uploaded_files:
        #     st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
        if st.button("Preview"):
            for uploaded_file in uploaded_files:
            # Convert uploaded file to an image for display
                image = Image.open(uploaded_file)

                # Resize the image (e.g., to 300x300 pixels)
                resized_image = image.resize((300, 300))

                # Display the resized image
                st.image(resized_image, caption=uploaded_file.name)
with col2:
        # Predict button
        if st.button("Predict"):
            results = {}
            with st.spinner("Making predictions..."):

                for uploaded_file in uploaded_files:
                    # Send image to backend for prediction
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(backend_url, files=files)

                    if response.status_code == 200:
                        predictions = response.json()
                        # Store predictions with the image name as key
                        results[uploaded_file.name] = predictions
                    else:
                        st.error(f"Error in prediction for {uploaded_file.name}: " + response.text)
            
            # Display predictions for each image
            st.write("### Predictions")
            for uploaded_file in uploaded_files:
                # Get the predictions for the current uploaded file
                image_name = uploaded_file.name
                predictions = results.get(image_name, {})
                
                with st.expander(f"View Result for image{image_name}"):
                    col3,col4 = st.columns([0.5,0.5])
                    # Display the input image
                    with col3:
                        image = Image.open(uploaded_file)  # Open the image file
                        resized_image = image.resize((400, 400))

                        # Display the resized image
                        st.image(resized_image, caption=uploaded_file.name)
                    # st.image(image, caption=image_name)
                    # Display the predictions for the current image
                    with col4:
                        if predictions:
                            for model_name, prediction in predictions.items():
                                st.write(f"- **{model_name}**: {prediction}")
                        else:
                            st.write("No predictions available for this image.")


            # for image_name, predictions in results.items():
            #     # st.write(f"**Image: {image_name}**")
            #     st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

            #     for model_name, prediction in predictions.items():
            #         st.write(f"- {model_name}: {prediction}")
with col1:
# Optionally, display a message if no file has been uploaded
    if not uploaded_files:
        st.info("Please upload one or more images to see predictions.")

            # # Send image to backend for prediction
            # files = {"file": uploaded_file.getvalue()}
            # response = requests.post(backend_url, files=files)

            # if response.status_code == 200:
            #     predictions = response.json() 

            #     # Display predictions for all models
            #     st.write("### Predictions from all models:")
            #     for model_name, prediction in predictions.items():
            #         st.write(f"**{model_name}**: {prediction}")
            # else:
            #     st.error("Error in prediction: " + response.text)
