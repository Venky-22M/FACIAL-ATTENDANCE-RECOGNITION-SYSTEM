import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize face recognizer & classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure attendance file exists
attendance_file = r"C:\OOSE PROJECT T8\attendance.xlsx"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_excel(attendance_file, index=False)

# Load trained faces
def train_faces():
    faces = []
    labels = []
    label_dict = {}
    label_count = 0
    faces_dir = r"C:\OOSE PROJECT T8\faces"  # Use raw string for Windows paths

    if not os.path.exists(faces_dir):
        print(f"Error: The folder {faces_dir} does not exist.")
        return {}

    for user in os.listdir(faces_dir):
        user_folder = os.path.join(faces_dir, user)

        if not os.path.exists(user_folder):
            print(f"Skipping {user_folder}: Folder not found.")
            continue

        label_dict[label_count] = user  # Map ID to username

        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                faces.append(img)
                labels.append(label_count)

        label_count += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        print("Training completed successfully.")
    else:
        print("Warning: No faces were found for training.")

    return label_dict

label_dict = train_faces()

def recognize_faces():
    if not label_dict:  # Check if training was successful
        print("Error: No trained data found. Train the model first.")
        return

    # Initialize camera
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return

    try:
        attendance = pd.read_excel(r"C:/OOSE PROJECT T8/attendance.xlsx")

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("Error: Unable to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            recognized_users = set()

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (150, 150))

                try:
                    label, confidence = recognizer.predict(face_resized)
                    if confidence < 60:
                        user = label_dict.get(label, "Unknown")
                        now = datetime.now()
                        current_date = now.strftime("%Y-%m-%d")
                        current_time = now.strftime("%H:%M:%S")

                        if user not in recognized_users:
                            recognized_users.add(user)
                            new_entry = pd.DataFrame([[user, current_date, current_time]], columns=["Name", "Date", "Time"])
                            attendance = pd.concat([attendance, new_entry], ignore_index=True)
                            print(f"Recognized: {user} at {current_date} {current_time}")

                except Exception as e:
                    print(f"Prediction error: {e}")

            cv2.imshow("Face Recognition", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save updated attendance
        attendance.to_excel("C:/OOSE PROJECT T8/attendance.xlsx", index=False)
        print("Attendance data saved successfully.")

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()