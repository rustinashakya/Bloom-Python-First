from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI(title="Cervical Cancer Detection API")

model = load_model("model/cervical_model.h5")

def preprocess(img: Image.Image):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img_array = preprocess(img)
    prediction = model.predict(img_array)[0][0]
    # In binary classification, classes are assigned alphabetically:
    # "Abnormal" = 0, "Normal" = 1
    # So prediction < 0.5 = Abnormal, prediction >= 0.5 = Normal
    label = "Abnormal" if prediction < 0.5 else "Normal"
    confidence = float(1 - prediction if label == "Abnormal" else prediction)
    return {"result": label, "confidence": confidence}
