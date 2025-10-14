from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

# ------------------------
# CORS configuration
# ------------------------
origins = [
    "https://potato-leaf-disease-prediction-35.netlify.app",  # exact frontend URL, no spaces or trailing slash
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # restrict to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Load model
# ------------------------
MODEL_PATH = "./Models/1.keras"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_names = ['Early Blight', 'Late Blight', 'Healthy']

# ------------------------
# Health check endpoint
# ------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am Rajiv."}

# ------------------------
# Helper: Read and preprocess image
# ------------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = np.array(image)
    return image


# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # batch size = 1

        # Get predictions
        predictions = model.predict(img_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# ------------------------
# Optional root endpoint
# ------------------------
@app.get("/")
async def root():
    return {"message": "Potato Leaf Disease API is running!"}

# ------------------------
# Note: On Render, start your app with:
# uvicorn API.main:app --host 0.0.0.0 --port $PORT
# ------------------------
