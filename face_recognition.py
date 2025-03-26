import cv2
import numpy as np
import mysql.connector

print("🔹 Starting Multi-Face Recognition...")

# ✅ Connect to MySQL Database
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Change if your MySQL username is different
    password="@Adhi143",  # Replace with your MySQL password
    database="face_recognition_db"
)
cursor = db.cursor()
print("✅ Connected to MySQL Database.")

# ✅ Load Face Detection Model (SSD)
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ Face Detection Model Loaded.")

# ✅ Load Face Recognition Model (SFace)
recognition_net = cv2.dnn.readNetFromONNX("face_recognition_sface_2021dec.onnx")
print("✅ Face Recognition Model Loaded.")

# Function to compute face embeddings
def get_face_embedding(face_image):
    blob = cv2.dnn.blobFromImage(face_image, scalefactor=1.0/255, size=(112, 112), mean=(0, 0, 0), swapRB=True, crop=False)
    recognition_net.setInput(blob)
    embedding = recognition_net.forward()
    embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding
    return embedding

# ✅ Load Faces from MySQL Instead of `known_faces/` Folder
known_faces_db = {}

cursor.execute("SELECT name, face_embedding FROM known_faces")
for name, embedding_str in cursor.fetchall():
    embedding_array = np.array(list(map(float, embedding_str.split(","))))
    known_faces_db[name] = embedding_array

print(f"✅ Loaded {len(known_faces_db)} known faces: {list(known_faces_db.keys())}")

# ✅ Function to Save New Faces to MySQL
def save_new_face(name, face_embedding):
    embedding_str = ",".join(map(str, face_embedding.flatten()))
    cursor.execute("INSERT INTO known_faces (name, face_embedding) VALUES (%s, %s)", (name, embedding_str))
    db.commit()
    print(f"✅ New face {name} added to MySQL!")

# ✅ Start Webcam
print("🔹 Starting webcam...")
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("❌ ERROR: Webcam could not be opened. Check your camera.")
    exit()

print("✅ Webcam started. Looking for faces...")

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("❌ ERROR: Could not read frame from webcam.")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect Faces
    face_net.setInput(blob)
    detections = face_net.forward()

    print(f"🔹 Detected {detections.shape[2]} faces.")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            print(f"✅ Face detected with {confidence:.2f} confidence.")

            # Extract Face
            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_embedding = get_face_embedding(face)

                # Compare with All Known Faces
                recognized_name = "Unknown"
                highest_similarity = 0.0

                for name, known_embedding in known_faces_db.items():
                    cosine_similarity = np.dot(face_embedding.flatten(), known_embedding.flatten()) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                    )
                    print(f"🔹 Comparing with {name}: Similarity Score = {cosine_similarity:.2f}")

                    if cosine_similarity > highest_similarity and cosine_similarity > 0.45:  # Adjust threshold
                        highest_similarity = cosine_similarity
                        recognized_name = name  # Set the recognized name

                print(f"🔹 Face Similarity: {highest_similarity:.2f} - Recognized as: {recognized_name}")

                # ✅ If Face is Unknown, Ask User for Name & Save to MySQL
                if recognized_name == "Unknown":
                    new_name = input("🚀 New Face Detected! Enter name: ")
                    save_new_face(new_name, face_embedding)
                    known_faces_db[new_name] = face_embedding  # Update local database

                # Draw Bounding Box & Name on Screen
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the Frame
    cv2.imshow("Multi-Face Recognition with MySQL", frame)

    # Press 'A' to Exit
    if cv2.waitKey(10) == ord("a"):
        break

# ✅ Release Resources
video_cap.release()
cv2.destroyAllWindows()
db.close()
