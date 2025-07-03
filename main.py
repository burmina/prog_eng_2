from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
import io
import os
import requests
from pathlib import Path

app = FastAPI(
    title="Plant Classification API",
    description="API for classifying plant images",
    version="1.0.0"
)

# Конфигурация
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1CJPORxXj-9nELB6M7aKLHQNEehk9mk4B'
MODEL_PATH = 'plant_classification_model.keras'
LABELS_PATH = 'class_labels.json'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            session = requests.Session()
            response = session.get(MODEL_URL, stream=True)
            
            # Сохраняем модель
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise
    else:
        print("Model already exists, using local copy")

# Загрузка модели и меток классов
try:
    download_model()
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
    
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    print("Class labels loaded")
except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise

@app.post("/predict/")
async def predict_plant(file: UploadFile = File(...)):
    # Проверка формата файла
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        # Чтение и предобработка изображения
        contents = await file.read()
        img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Предсказание
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "class": class_labels[predicted_class_idx],
            "confidence": confidence,
            "all_predictions": dict(zip(class_labels, predictions[0].tolist()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Plant Classification API - Use /predict/ endpoint"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)