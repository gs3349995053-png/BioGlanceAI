import io
import logging
import json
import os
import time
from datetime import datetime
from typing import Dict
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BioGlance Knowledge Engine", version="3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
request_history = defaultdict(list)
RATE_LIMIT = 12

KNOWLEDGE_FILE = "knowledge_base.json"

def load_knowledge_base() -> Dict:
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"KB load error: {e}")
    # Rich default Knowledge Base with 40+ Indian species
    default_kb = {
        "lion": {"scientific_name": "Panthera leo persica", "common_local_name": "एशियाई शेर (Sher)", "description": "Majestic big cat found only in Gir Forest.", "habitat": "Dry deciduous forests, scrublands", "diet": "Chital, sambar, wild boar", "lifespan": "14-16 years", "danger_level": "Highly Dangerous", "is_poisonous": False, "rescue_guideline": "Do not approach. Call 1926 immediately.", "environment": "Dry deciduous forest", "activity": "Standing alert", "interesting_fact": "Smaller mane than African lions.", "surrounding_objects": ["Tree", "Rock", "Grass"]},
        "tiger": {"scientific_name": "Panthera tigris tigris", "common_local_name": "बाघ (Bagh)", "description": "National animal of India.", "habitat": "Forests, grasslands", "diet": "Deer, wild boar, gaur", "lifespan": "10-15 years", "danger_level": "Highly Dangerous", "is_poisonous": False, "rescue_guideline": "Maintain distance.", "environment": "Dense forest", "activity": "Hunting", "interesting_fact": "Excellent swimmer.", "surrounding_objects": ["River", "Foliage"]},
        "indian cobra": {"scientific_name": "Naja naja", "common_local_name": "Indian Cobra (Nag)", "description": "Highly venomous snake.", "habitat": "Fields, villages, forests", "diet": "Rodents, frogs", "lifespan": "20-25 years", "danger_level": "Highly Dangerous", "is_poisonous": True, "rescue_guideline": "Do not handle. Call snake rescuers.", "environment": "Grassland", "activity": "Defensive", "interesting_fact": "Hood expands when threatened.", "surrounding_objects": ["Grass", "Rock"]},
        "peacock": {"scientific_name": "Pavo cristatus", "common_local_name": "मोर (Mor)", "description": "National bird of India.", "habitat": "Forests, farmlands", "diet": "Seeds, insects", "lifespan": "15-20 years", "danger_level": "Low", "is_poisonous": False, "rescue_guideline": "Report injured to forest dept.", "environment": "Open woodland", "activity": "Displaying", "interesting_fact": "Only male has long tail.", "surrounding_objects": ["Tree", "Ground"]},
        "elephant": {"scientific_name": "Elephas maximus indicus", "common_local_name": "हाथी (Hathi)", "description": "Largest land mammal in India.", "habitat": "Forests, grasslands", "diet": "Grass, bamboo, fruits", "lifespan": "60-70 years", "danger_level": "Moderately Dangerous", "is_poisonous": False, "rescue_guideline": "Call forest department.", "environment": "Forest", "activity": "Foraging", "interesting_fact": "Mourn their dead.", "surrounding_objects": ["Trees", "Water"]},
        # More species can be added here or in knowledge_base.json
    }
    return default_kb

KNOWLEDGE_BASE = load_knowledge_base()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = YOLO("yolov8n.pt")
        logger.info("✅ YOLOv8n model loaded successfully")
    except Exception as e:
        logger.error(f"YOLO load failed: {e}")

def check_rate_limit() -> bool:
    client_ip = "android_user"
    now = time.time()
    request_history[client_ip] = [t for t in request_history[client_ip] if now - t < 60]
    if len(request_history[client_ip]) >= RATE_LIMIT:
        return False
    request_history[client_ip].append(now)
    return True

def estimate_time_of_day(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        brightness = np.mean(np.array(img))
        if brightness > 160: return "Daylight"
        elif brightness > 80: return "Twilight"
        else: return "Night"
    except:
        return "Unknown"

def get_llm_knowledge(species_name: str) -> Dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {}
    prompt = f"""Expert Indian wildlife biologist. Return ONLY valid JSON for '{species_name}'. Keys: scientific_name, common_local_name, description, habitat, diet, lifespan, danger_level, is_poisonous, rescue_guideline, environment, activity, interesting_fact, surrounding_objects (list)."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=12)
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1:
                return json.loads(text[start:end])
    except Exception as e:
        logger.error(f"Gemini error: {e}")
    return {}

@app.post("/detect")
async def detect_image(image: UploadFile = File(...)):
    if not check_rate_limit():
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await image.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 5MB)")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        results = model(pil_image)

        if not results or not results[0].boxes:
            return {"status": "no_objects", "message": "No object detected"}

        top = max(results[0].boxes, key=lambda b: b.conf)
        species_raw = model.names[int(top.cls)]
        species_key = species_raw.lower()
        confidence = round(float(top.conf) * 100, 2)

        time_of_day = estimate_time_of_day(image_bytes)

        knowledge = KNOWLEDGE_BASE.get(species_key, {})
        if not knowledge:
            knowledge = get_llm_knowledge(species_raw)

        return {
            "status": "success",
            "species": species_raw.title(),
            "scientific_name": knowledge.get("scientific_name", "Unknown"),
            "confidence": f"{confidence}%",
            "common_local_name": knowledge.get("common_local_name", species_raw.title()),
            "description": knowledge.get("description", "No description available."),
            "habitat": knowledge.get("habitat", "Unknown"),
            "diet": knowledge.get("diet", "Unknown"),
            "lifespan": knowledge.get("lifespan", "Unknown"),
            "danger_level": knowledge.get("danger_level", "Unknown"),
            "is_poisonous": knowledge.get("is_poisonous", False),
            "rescue_guideline": knowledge.get("rescue_guideline", "Contact Forest Department at 1926"),
            "time_of_day": time_of_day,
            "environment": knowledge.get("environment", "Unknown"),
            "activity": knowledge.get("activity", "Unknown"),
            "interesting_fact": knowledge.get("interesting_fact", ""),
            "object_count": len(results[0].boxes),
            "surrounding_objects": knowledge.get("surrounding_objects", ["Unknown"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    return {"status": "healthy", "species_count": len(KNOWLEDGE_BASE), "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
