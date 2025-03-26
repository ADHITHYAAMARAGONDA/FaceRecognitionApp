from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import cv2
import numpy as np
import mysql.connector

app = Flask(__name__)

# ✅ Function to Connect to Railway MySQL Database
def connect_db():
    return mysql.connector.connect(
        host="shinkansen.proxy.rlwy.net",
        user="root",
        password="LvJzTXIxEJDOgrNISquawcVlhVZwcrtv",
        database="railway",
        port=53399
    )

db = connect_db()
cursor = db.cursor()
print("✅ Connected to Railway MySQL Database.")

# ✅ Load Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

known_faces_db = load_known_faces()
print(f"✅ Loaded {len(known_faces_db)} known faces: {list(known_faces_db.keys())}")

# ✅ Route to Process Webcam Frames (Automatic Face Recognition)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return "No frame received", 400

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    recognized_name = "Unknown"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.shape[0] > 0 and face.shape[1] > 0:
            face_embedding = get_face_embedding(face)

            # Compare with known faces
            highest_similarity = 0.0
            for name, known_embedding in known_faces_db.items():
                cosine_similarity = np.dot(face_embedding, known_embedding) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                )
                if cosine_similarity > highest_similarity and cosine_similarity > 0.55:
                    highest_similarity = cosine_similarity
                    recognized_name = name  # Set recognized name

            # Draw bounding box and name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# ✅ Route to Add New Face
@app.route('/add_face', methods=['POST'])
def add_face():
    global known_faces_db
    name = request.form['name']

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detect Face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.shape[0] > 0 and face.shape[1] > 0:
            face_embedding = get_face_embedding(face)
            face_embedding_str = ",".join(map(str, face_embedding))

            # Insert into Database
            cursor.execute("INSERT INTO known_faces (name, face_embedding) VALUES (%s, %s)", (name, face_embedding_str))
            db.commit()

            # Update Local Known Faces
            known_faces_db = load_known_faces()
            return jsonify({"message": f"✅ {name} added successfully!"})

    return jsonify({"error": "❌ Face not detected."})

# ✅ Route for Admin Panel (View & Delete Faces)
@app.route('/admin')
def admin():
    cursor.execute("SELECT id, name FROM known_faces")
    faces = cursor.fetchall()
    return render_template('admin.html', faces=faces)

# ✅ Route to Delete a Face
@app.route('/delete_face', methods=['POST'])
def delete_face():
    face_id = request.form['face_id']
    cursor.execute("DELETE FROM known_faces WHERE id = %s", (face_id,))
    db.commit()
    return redirect(url_for('admin'))

# ✅ Flask Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
