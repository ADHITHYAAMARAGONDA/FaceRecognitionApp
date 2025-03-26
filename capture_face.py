import cv2

# Start webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Capture Your Face - Press 'S' to Save", frame)

    # Press 'S' to save the image
    if cv2.waitKey(1) & 0xFF == ord("s"):
        cv2.imwrite("known_face.jpg", frame)  # Save face image
        print("✅ Face saved as known_face.jpg")
        break

    # Press 'Q' to quit without saving
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()
