# 🚀 AI Sentiment Analysis System using BiLSTM

A real-world Deep Learning and NLP project that performs real-time sentiment analysis on user text using a BiLSTM neural network trained on 1.6 million Twitter/X tweets.

This project includes:

- Deep Learning NLP Model
- FastAPI Backend
- Interactive Frontend
- Real-time Prediction
- Cloudflare Live Deployment
- Industry-Level Workflow

---

# 📌 Features

✅ Real-time Sentiment Prediction  
✅ NLP Text Preprocessing  
✅ BiLSTM Deep Learning Architecture  
✅ Confidence Score Prediction  
✅ FastAPI REST API  
✅ Modern Responsive Frontend  
✅ Cloudflare Tunnel Live Hosting  
✅ Kaggle GPU/TPU Training Support  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend & AI |
| TensorFlow/Keras | Deep Learning |
| BiLSTM | Sequence Modeling |
| NLP | Text Processing |
| FastAPI | API Backend |
| HTML/CSS/JavaScript | Frontend |
| Cloudflare Tunnel | Public Hosting |
| Kaggle | Model Training |

---

# 📊 Dataset

## Sentiment140 Dataset

- 1.6 Million real Twitter/X tweets
- Real-world noisy text data
- Binary sentiment classification

Dataset Link:

https://www.kaggle.com/datasets/kazanova/sentiment140

---

# 🏗 Project Architecture

```text
Frontend (HTML/CSS/JS)
          ↓
FastAPI Backend
          ↓
BiLSTM Sentiment Model
          ↓
Prediction Response
```

---

# 📁 Project Structure

```text
sentiment-analysis/
│
├── main.py
├── index.html
├── sentiment_model.h5
├── tokenizer.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPO_URL
```

---

## 2. Open Project Folder

```bash
cd sentiment-analysis
```

---

## 3. Create Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

# 🌐 Run Frontend

Use VS Code Live Server extension or open `index.html`.

---

# ☁️ Cloudflare Live Deployment

## Backend Tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

---

## Frontend Tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:5500
```

---

# 🔥 API Endpoints

## Home API

```http
GET /
```

---

## Health API

```http
GET /health
```

---

## Prediction API

```http
POST /predict
```

---

# 📥 Example Request

```json
{
    "text": "I love this AI project"
}
```

---

# 📤 Example Response

```json
{
    "input_text": "I love this AI project",
    "cleaned_text": "love ai project",
    "sentiment": "Positive Sentiment",
    "confidence_score": 0.9245
}
```

---

# 🧹 NLP Preprocessing

The project performs:

- Lowercasing
- URL removal
- Mention removal
- Hashtag cleaning
- Punctuation removal
- Stopword removal
- Lemmatization

---

# 🧠 Model Architecture

```text
Embedding Layer
        ↓
BiLSTM Layer
        ↓
GlobalAveragePooling
        ↓
Dropout
        ↓
Dense Layer
        ↓
Sigmoid Output
```

---

# 📈 Training Features

✅ Tokenization  
✅ Sequence Padding  
✅ Early Stopping  
✅ Accuracy Visualization  
✅ Confusion Matrix  
✅ Classification Report  

---

# 🧪 Example Test Inputs

## Positive

```text
This AI project is absolutely amazing
```

---

## Negative

```text
The application is terrible and frustrating
```

---

## Mixed

```text
The food was delicious but delivery was slow
```

---

# 📸 Screenshots

Add screenshots here:

- Frontend UI
- Prediction Result
- Training Graph
- Confusion Matrix
- Swagger API

---

# 🚀 Future Improvements

- Attention Mechanism
- Transformer Models
- BERT Integration
- Docker Deployment
- HuggingFace Deployment
- Real-Time Tweet Analysis
- Database Integration
- Authentication System

---

# 💡 Learning Outcomes

This project demonstrates:

- Deep Learning
- NLP Pipelines
- Sequence Modeling
- API Development
- AI Deployment
- Production Workflow
- Frontend + Backend Integration

---

# 👨‍💻 Author

## Kanha Patidar

B.Tech CSIT Student | AI & Deep Learning Enthusiast

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
⭐ Fork the project  
⭐ Connect on LinkedIn  

---

# 📜 License

This project is licensed under the MIT License.
