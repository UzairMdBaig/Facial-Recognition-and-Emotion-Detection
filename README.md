# 😃 Facial Recognition & Emotion Detection (Webcam + ONNX)

This project demonstrates an end-to-end facial recognition and emotion detection system built with:

**Models:**

- **Face Recognition**: InceptionResnetV1 (pretrained on VGGFace2) via facenet-pytorch

- **Emotion Detection**: ONNX model (emotion_model.onnx)

**Backend**: FastAPI serving recognition + emotion inference using ONNX Runtime & PyTorch.

**Frontend**: React app (with react-webcam) streaming live predictions.

**Deployment**: Fully containerized with Docker & orchestrated using Docker Compose with Nginx as reverse proxy.

---

## 📂 Project Structure (Main folders/files)

```bash
.
├── API/                         # FastAPI backend
│   ├── model/
│   │   └── emotion_model.onnx    # ONNX model for emotion detection
│   ├── embeddings/               # Stored average embeddings (e.g., Ronnie, Jay)
│   │   ├── ronnie_colman_embedding.npy
│   │   └── jay_cutler_embedding.npy
│   ├── Dockerfile
│   ├── main.py                   # API logic
│   └── requirements.txt
│
├── frontend/                     # React frontend
│   ├── Dockerfile
│   ├── nginx.conf                # Reverse proxy config
│   └── src/
│       └── App.jsx               # Webcam + API calls
│   
│
├── docker-compose.yml            # Multi-service setup
|
└── README.md
```

## 🚀 Models & Processing
**Face Recognition**
- Detects face using MTCNN
- Extracts embeddings using **InceptionResnetV1** (pretrained VGGFace2)
- Compares embeddings against stored average embeddings (ronnie, jay)
- Threshold-based matching → Ronnie Coleman, Jay Cutler, or Unknown

**Emotion Detection**
- Face cropped using MTCNN (48×48 grayscale preprocessing)
- Passed into ONNX model, ie **InceptionResnetV1** (pretrained on Fer2013 dataset)
- Predicts one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## ▶️ Running the Project

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Facial-Recognition-and-Emotion-Detection.git
cd facial-recognition-emotion-detection
```

**2. Build & Run with Docker Compose**
```bash
docker-compose up --build
```

**3. Access the frontend**
```bash
👉 http://localhost:60
```

---

## 🌐 API Endpoints

**POST /predict**

- **Input**: Image file (multipart/form-data)

- **Output**: JSON response

```json
{
  "ID": "Ronnie Coleman",
  "Emotion": "Happy"
}
```

## 🛠 Tech Stack

- **Backend**: FastAPI, ONNX Runtime, facenet-pytorch, OpenCV, NumPy

- **Frontend**: React, React-Webcam, TailwindCSS

- **Proxy**: Nginx

- **Deployment**: Docker, Docker Compose
