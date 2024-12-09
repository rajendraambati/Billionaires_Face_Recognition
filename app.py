import os
import torch
import numpy as np
import cv2
import pickle
from PIL import Image
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings("ignore")

# Initialize device
device = torch.device('cpu')

# Load MTCNN for face detection using the specified weights
mtcnn_model_path = 'inception_resnet_v1_mtcnn.pth'  # Update this path to your MTCNN model
mtcnn = MTCNN(keep_all=True, device=device)
mtcnn.load_state_dict(torch.load(mtcnn_model_path, map_location=device))
#st.write("MTCNN model loaded successfully.")

# Load Inception Resnet V1 for face recognition using the specified weights
recognition_model_path = 'inception_resnet_v1_weights.pth'  # Update this path to your recognition model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model.load_state_dict(torch.load(recognition_model_path, map_location=device))
#st.write("Inception ResNet V1 model loaded successfully.")

# Load the face database from a pickle file
with open('face_database.pkl', 'rb') as f:  # Update this path to your pickle file
    face_database = pickle.load(f)
#st.write("Face database loaded successfully.")

def extract_face_embeddings(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is not None:
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = np.transpose(face, (2, 0, 1)) / 255.0
            face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)
            faces.append(face_tensor)
        if len(faces) > 0:
            faces = torch.cat(faces)
            embeddings = model(faces).detach().cpu().numpy()
            return boxes, embeddings
    return None, None

def recognize_faces(boxes, embeddings):
    """Recognize faces based on extracted embeddings."""
    recognized_names = []
    for face_embedding in embeddings:
        min_distance = float('inf')
        best_match = "Unknown"
        
        # Compare with known embeddings in the database
        for name, known_embeddings in face_database.items():
            distances = [np.linalg.norm(face_embedding - known_emb) for known_emb in known_embeddings]
            current_min_distance = min(distances)

            if current_min_distance < min_distance:
                min_distance = current_min_distance
                best_match = name

        # Assign label only if there's a significant gap between the closest match and a threshold
        if min_distance < 0.9:  # You can adjust this threshold based on your needs
            recognized_names.append(best_match)
        else:
            recognized_names.append("Unknown")

    return recognized_names

def main():
    st.title("Face Recognition App")
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        #st.image(image_np, caption='Uploaded Image', use_column_width=True)

        boxes, embeddings = extract_face_embeddings(image_np)
        if embeddings is not None:
            st.write("Face embeddings extracted successfully!")
            
            recognized_names = recognize_faces(boxes, embeddings)

            for box, name in zip(boxes, recognized_names):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                
                # Put label above the bounding box
                cv2.putText(image_np, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the processed image with bounding boxes and labels
            st.image(image_np, caption='Processed Image with Detected Faces', use_column_width=True)
        else:
            st.write("No faces detected.")

if __name__ == "__main__":
    main()
