from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import mysql.connector
import os

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Function to Connect to MySQL Database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@Adhi143",  # Replace with your MySQL password
        database="face_recognition_db"
    )

# ✅ Establish Initial Database Connection
db = connect_db()
cursor = db.cursor()
print("✅ Connected to MySQL Database.")

# ✅ Load Face Detection Model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ Face Detection Model Loaded.")

# ✅ Load Face Recognition Model
recognition_net = cv2.dnn.readNetFromONNX("face_recognition_sface_2021dec.onnx")
print("✅ Face Recognition Model Loaded.")

# ✅ Function to Compute Face Embeddings
def get_face_embedding(face_image):
    blob = cv2.dnn.blobFromImage(face_image, scalefactor=1.0/255, size=(112, 112), mean=(0, 0, 0), swapRB=True, crop=False)
    recognition_net.setInput(blob)
    embedding = recognition_net.forward()
    embedding = embedding.flatten() / np.linalg.norm(embedding)  # Normalize embedding
    return embedding

# ✅ Load Known Faces from MySQL
def load_known_faces():
    known_faces_db = {}
    cursor.execute("SELECT name, face_embedding FROM known_faces")
    for name, embedding_str in cursor.fetchall():
        embedding_array = np.array(list(map(float, embedding_str.split(","))))
        known_faces_db[name] = embedding_array
    return known_faces_db

# ✅ Global Variable to Store Known Faces
known_faces_db = load_known_faces()
print(f"✅ Loaded {len(known_faces_db)} known faces: {list(known_faces_db.keys())}")

# ✅ Function to Start Webcam & Detect Faces
def generate_frames():
    video_cap = cv2.VideoCapture(0)
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame size for faster processing
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        success, frame = video_cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Detect Faces
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Increased threshold to reduce false positives
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract Face
                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_embedding = get_face_embedding(face)

                    # Compare with Known Faces
                    recognized_name = "Unknown"
                    highest_similarity = 0.0

                    for name, known_embedding in known_faces_db.items():
                        cosine_similarity = np.dot(face_embedding, known_embedding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                        )

                        if cosine_similarity > highest_similarity and cosine_similarity > 0.55:  # Adjusted threshold
                            highest_similarity = cosine_similarity
                            recognized_name = name  # Set the recognized name

                    # Draw Bounding Box & Name
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode Frame for Streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Optimize JPEG size
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ✅ Flask Route for Video Stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Flask Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Flask Route for Admin Panel (View & Delete Faces)
@app.route('/admin')
def admin():
    cursor.execute("SELECT id, name FROM known_faces")
    faces = cursor.fetchall()
    return render_template('admin.html', faces=faces)

# ✅ Flask Route to Delete a Face
@app.route('/delete_face', methods=['POST'])
def delete_face():
    face_id = request.form['face_id']
    cursor.execute("DELETE FROM known_faces WHERE id = %s", (face_id,))
    db.commit()
    return redirect(url_for('admin'))

# ✅ Flask Route to Manually Add a New Face
@app.route('/add_face', methods=['POST'])
def add_face():
    global known_faces_db
    name = request.form['name']
    
    # Capture and Save New Face
    video_cap = cv2.VideoCapture(0)
    success, frame = video_cap.read()
    video_cap.release()

    if success:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Detect Faces
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract Face
                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_embedding = get_face_embedding(face)
                    face_embedding_str = ",".join(map(str, face_embedding))

                    # Insert into Database
                    cursor.execute("INSERT INTO known_faces (name, face_embedding) VALUES (%s, %s)", (name, face_embedding_str))
                    db.commit()

                    # Update Local Known Faces
                    known_faces_db = load_known_faces()
                    return f"✅ {name} added successfully! <a href='/'>Go Back</a>"
    
    return "❌ Face not detected. <a href='/'>Try Again</a>"

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
