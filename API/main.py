from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
import onnxruntime as ort
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ed_model = ort.InferenceSession("model/emotion_model.onnx")
fr_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn_fr = MTCNN(image_size=160, margin=0)
mtcnn_ed = MTCNN(image_size=48, margin=0)


ronnie_avg_embedding = np.load('embeddings/ronnie_colman_embedding.npy')
jay_avg_embedding = np.load('embeddings/jay_cutler_embedding.npy')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def emotion_detection_preprocess(image):
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, -1)

    return image

def find_ID(embedding, threshold=0.8):
    dist_ronnie = np.linalg.norm(embedding - ronnie_avg_embedding)
    dist_jay= np.linalg.norm(embedding - jay_avg_embedding)

    if dist_ronnie < threshold:
        return "Ronnie Coleman" 
    elif dist_jay < threshold:
        return "Jay Cutler" 
    else:
        return "Unknown"
    

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_fr = mtcnn_fr(image)
    if crop_fr is None:
        return {"ID": "No face detected", "Emotion": "No emotion detected"}
    result_fr = fr_model(crop_fr.unsqueeze(0)).detach().numpy()
    id = find_ID(result_fr)

    crop_ed = mtcnn_ed(image)
    crop_ed = emotion_detection_preprocess(crop_ed)
    result_ed = ed_model.run(None, {'input': crop_ed})[0][0]  
    emotion_idx = np.argmax(result_ed)
    emotion = emotions[emotion_idx]

    return {"ID": id, "Emotion": emotion}