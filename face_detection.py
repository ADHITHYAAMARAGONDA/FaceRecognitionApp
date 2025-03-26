import cv2
import numpy as np

# Load pre-trained deep learning model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Start webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Convert frame to a blob (for deep learning model)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    # Pass the blob through the model
    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop through detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only proceed if confidence is above 50%
        if confidence > 0.5:
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the face bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Deep Learning Face Detection", frame)

    # Exit when "a" is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
