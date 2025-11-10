from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()


# CORS configuration
origins = [
    "https://potato-leaf-disease-prediction-35.netlify.app",  
     "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./Models/1.keras" 
model = load_model(MODEL_PATH) 
class_names = ['Early Blight', 'Late Blight', 'Healthy'] 

# verify that the FastAPI server is running.
@app.get("/ping") 
async def ping():
    return "Hello, I am Rajiv." 

def read_file_as_image(data) -> np.ndarray: 
    image = Image.open(BytesIO(data)) 
    image = np.array(image) 
    return image 
    
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

@app.get("/")
async def root():
    return {"message": "Potato Leaf Disease API is running!"}

# uvicorn API.main:app --host 0.0.0.0 --port $PORT
