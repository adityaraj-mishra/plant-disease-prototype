from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image


app = FastAPI()

MODEL = tf.keras.models.load_model("../model/2")
CLASS_NAMES = ["Early Blight", "Late Blight","Healthy"]



@app.get("/ping")
async def ping():
    return "Hello, I am working"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.argmax(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence*100)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)