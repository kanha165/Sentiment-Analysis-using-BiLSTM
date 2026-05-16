from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fastapi.middleware.cors import CORSMiddleware

import pickle
import re
import string
import nltk
import gdown
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# DOWNLOAD NLTK FILES
# ===============================

nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="BiLSTM Sentiment Analysis",
    version="1.0"
)

# ===============================
# STATIC + TEMPLATE
# ===============================

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ===============================
# CORS
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# DOWNLOAD MODEL
# ===============================

MODEL_PATH = "sentiment_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1PjMKDtoSHHRoCCuVfUGS7Zvdrl_rJDu-"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ==========================================
# 🔥 KERAS BATCH_SHAPE ERROR FIX PATCH
# ==========================================
from keras.layers import InputLayer
original_input_layer_init = InputLayer.__init__

def patched_input_layer_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['shape'] = kwargs.get('shape', kwargs['batch_shape'][1:])
        kwargs.pop('batch_shape', None)
    original_input_layer_init(self, *args, **kwargs)

InputLayer.__init__ = patched_input_layer_init
# ==========================================

# ===============================
# LOAD MODEL
# ===============================

model = load_model(MODEL_PATH)

# ===============================
# LOAD TOKENIZER
# ===============================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ===============================
# NLP TOOLS
# ===============================

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

MAX_LEN = 100

# ===============================
# REQUEST MODEL
# ===============================

class SentimentRequest(BaseModel):
    text: str

# ===============================
# CLEAN TEXT FUNCTION
# ===============================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'#', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = text.strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# ===============================
# FRONTEND PAGE
# ===============================

@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ===============================
# HEALTH API
# ===============================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }

# ===============================
# PREDICTION API
# ===============================

@app.post("/predict")
def predict(data: SentimentRequest):

    cleaned_text = clean_text(data.text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN
    )

    prediction = model.predict(padded)[0][0]

    sentiment = (
        "Positive Sentiment"
        if prediction > 0.5
        else "Negative Sentiment"
    )

    confidence = float(prediction)

    return {
        "input_text": data.text,
        "cleaned_text": cleaned_text,
        "sentiment": sentiment,
        "confidence_score": round(confidence, 4)
    }
