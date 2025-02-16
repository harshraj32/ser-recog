from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
import librosa
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
MODEL_PATH = "best_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to preprocess audio
def preprocess_audio(file: bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file), sr=22050)  # Load audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCCs
        
        # Ensure the second dimension (time steps) is at least 130
        if mfccs.shape[1] < 130:
            pad_width = 130 - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :130]  # Trim to 130 time steps

        # Reshape to (1, 40, 130, 1) -> Matching model input
        mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
        mfccs = np.expand_dims(mfccs, axis=0)   # Add batch dimension

        return mfccs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

class_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        audio_data = await file.read()
        processed_audio = preprocess_audio(audio_data)
        prediction = model.predict(processed_audio)
        class_idx = np.argmax(prediction)
        
        # Map class index to label
        class_label = class_labels[str(class_idx + 1).zfill(2)]  # Adding 1 to match the class labels
        return {
            "class": class_label,
            "confidence": float(np.max(prediction)),
            "predictions": prediction.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

