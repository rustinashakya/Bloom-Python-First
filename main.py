from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model/cervical_mobilenetv2.h5")


# img_path = "dataset_binary/test/Normal/209566205-209566289-001.BMP"
# img_path = "dataset_binary/test/Abnormal/149315775-149315790-003.BMP"
# img_path = "dataset_binary/test/Normal/157181569-157181599-001.BMP"
img_path = "dataset_binary/train/Abnormal/148495491-148495504-001.BMP"
# img_path = "dataset_binary/test/Normal/157224412-157224429-001.BMP"
# img_path = "dataset_binary/test/Abnormal/149317114-149317152-004.BMP"
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0][0]
# In binary classification, classes are assigned alphabetically:
# "Abnormal" = 0, "Normal" = 1
# So prediction < 0.5 = Abnormal, prediction >= 0.5 = Normal
result = "Abnormal" if pred < 0.5 else "Normal"
confidence = round(float(1 - pred if result == "Abnormal" else pred), 2)

print(f"🧬 Prediction: {result} ({confidence*100}% confidence)")
