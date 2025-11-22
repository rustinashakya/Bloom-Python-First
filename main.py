"""
Simple Cervical Cancer Detection API
Just copy this file and run it

Install:
    pip install fastapi uvicorn python-multipart tensorflow opencv-python pillow

Run:
    python simple_api.py

Test:
    Open browser: http://localhost:8000
    Or: curl -X POST "http://localhost:8000/predict" -F "file=@image.bmp"
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import cv2
from typing import List
import uvicorn

# Load your model
MODEL_PATH = 'final_model_combined.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Settings
IMG_SIZE = 224
THRESHOLD = 0.55

app = FastAPI(title="Cervical Cancer Detection")

@app.get("/", response_class=HTMLResponse)
def home():
    """Home page with upload form"""
    return """
    <html>
        <head><title>Cervical Cancer Detection</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>Cervical Cancer Detection API</h1>
            
            <h2>Single Image</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Predict</button>
            </form>
            
            <h2>Multiple Images</h2>
            <form action="/predict/batch" method="post" enctype="multipart/form-data">
                <input type="file" name="files" accept="image/*" multiple required>
                <button type="submit">Predict Batch</button>
            </form>
            
            <hr>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """Predict single image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        score = float(model.predict(img, verbose=0)[0][0])
        prediction = "Abnormal" if score >= THRESHOLD else "Normal"
        confidence = score * 100 if prediction == "Abnormal" else (1 - score) * 100
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "score": round(score, 4),
            "confidence": round(confidence, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict multiple images"""
    results = []
    
    for file in files:
        try:
            # Read and preprocess
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            score = float(model.predict(img, verbose=0)[0][0])
            prediction = "Abnormal" if score >= THRESHOLD else "Normal"
            confidence = score * 100 if prediction == "Abnormal" else (1 - score) * 100
            
            results.append({
                "filename": file.filename,
                "prediction": prediction,
                "score": round(score, 4),
                "confidence": round(confidence, 2)
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    # Summary
    normal = sum(1 for r in results if r.get("prediction") == "Normal")
    abnormal = sum(1 for r in results if r.get("prediction") == "Abnormal")
    
    return {
        "total": len(results),
        "normal": normal,
        "abnormal": abnormal,
        "results": results
    }

if __name__ == "__main__":
    print("Starting API server...")
    print("Open browser: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)