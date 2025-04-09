# Face Recognition App

## Overview
This project is a real-time facial recognition application built using Python. It utilizes OpenCV and deep learning techniques to detect and recognize faces from live webcam input. The system is designed for fast and accurate recognition and is optimized for real-time performance.

## Features
- Real-time face detection and recognition using OpenCV
- High-accuracy model using ONNX for face embeddings
- Live webcam integration for seamless facial scanning
- Cloud-based MySQL database integration for face data storage
- Flask API integration for deployment
- Web-based interface using JavaScript and WebRTC
- Deployment on Render with 99.9% uptime

## Project Structure
```
FaceRecognitionApp/
├── app.py                          # Flask backend API
├── recognize_faces.py              # Face recognition logic
├── train_model.py                  # Face encoding and training logic
├── templates/                      # HTML files for UI
├── static/                         # CSS/JS files
├── database/                       # MySQL schema (if provided)
├── utils/                          # Helper functions
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Technologies Used
- Python
- OpenCV
- ONNX
- Flask
- MySQL (hosted on Railway)
- JavaScript
- WebRTC
- Render (deployment)
- Git, GitHub

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/ADHITHYAAMARAGONDA/FaceRecognitionApp.git
   ```
2. Navigate to the project directory:
   ```
   cd FaceRecognitionApp
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Configure database connection in `app.py` or `.env`.

5. Run the application:
   ```
   python app.py
   ```
6. Access the app in your browser at `http://localhost:5000`.

## Output / Results
- Real-time face recognition from webcam
- Displays matched names from the database
- Stores new face data to MySQL if not recognized

## Learnings
- Implemented a production-ready face recognition system
- Optimized model inference using ONNX
- Built and deployed Flask REST APIs
- Integrated frontend (JavaScript) with backend APIs
- Learned cloud deployment using Render and Railway

## Future Improvements
- Add user authentication system
- Add role-based access (admin/user)
- Build a dashboard for face analytics
- Improve mobile responsiveness of frontend

## Acknowledgements
- OpenCV documentation
- Railway & Render hosting platforms
- ONNX Model Zoo

## License
This project is open-source and available under the MIT License.
