from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
import joblib
import cv2
import os

# Load models at startup
model_dir = os.path.join(os.getcwd(), 'models')
loaded_models = {
    'lr': joblib.load(os.path.join(model_dir, 'lr_model.pkl')),
    'rc': joblib.load(os.path.join(model_dir, 'rc_model.pkl')),
    'rf': joblib.load(os.path.join(model_dir, 'rf_model.pkl')),
    'gb': joblib.load(os.path.join(model_dir, 'gb_model.pkl'))
}

# Initialize MediaPipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize FastAPI
app = FastAPI()

# def extract_keypoints(image):
#     """Extract keypoints using MediaPipe."""
#     results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     if not results.pose_landmarks:
#         raise ValueError("No pose landmarks detected.")
#     keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
#                           for lm in results.pose_landmarks.landmark]).flatten()
#     return keypoints

import mediapipe as mp
import numpy as np
import pandas as pd
import cv2

# Initialize MediaPipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(image, class_name="unknown"):
    """Extract pose and face keypoints using MediaPipe, return as DataFrame."""
    
    # Process image with MediaPipe
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Check if landmarks are present
    if not results.pose_landmarks or not results.face_landmarks:
        raise ValueError("No pose or face landmarks detected.")
    
    # Calculate the number of coordinates (pose + face landmarks)
    num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
    
    # Define column names for DataFrame
    landmarks = ['class'] + [
        f"{axis}{val}" for val in range(1, num_coords + 1) for axis in ['x', 'y', 'z', 'v']
    ]
    
    # Extract Pose landmarks
    pose = results.pose_landmarks.landmark
    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())
    
    # Extract Face landmarks
    face = results.face_landmarks.landmark
    face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())
    
    # Concatenate pose and face data
    row = [class_name] + pose_row + face_row
    
    # Initialize DataFrame and add row
    df = pd.DataFrame([row], columns=landmarks)
    
    # Ensure DataFrame has 2005 columns
    if df.shape[1] != 2005:
        raise ValueError(f"Expected 2005 columns, but got {df.shape[1]} columns.")
    
    return df


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read image
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Extract keypoints
#         keypoints = extract_keypoints(image).reshape(1, -1)
        
#         # Get predictions from all models
#         predictions = {algo: int(model.predict(keypoints)[0]) 
#                        for algo, model in loaded_models.items()}

#         return JSONResponse(content=predictions)

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Extract keypoints into a DataFrame
        keypoints_df = extract_keypoints(image, class_name="unknown")
        
        # Ensure the DataFrame has the correct columns
        if keypoints_df.shape[1] != 2005:
            raise ValueError(f"Expected 2005 columns, but got {keypoints_df.shape[1]} columns.")
        
        # Drop the 'class' column to use only keypoints for prediction
        keypoints = keypoints_df.drop(columns=["class"])

        # # Get predictions from all models
        # predictions = {algo: int(model.predict(keypoints)[0]) 
        #                for algo, model in loaded_models.items()}
        # print(type(predictions))

        predictions = {algo: model.predict(keypoints)[0] for algo, model in loaded_models.items()}

        return JSONResponse(content=predictions)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))