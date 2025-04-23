import cv2
import os

def register_user(name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    count = 0

    # Create folder for the user
    user_folder = os.path.join("faces", name)
    os.makedirs(user_folder, exist_ok=True)

    print("Capturing face images for user:", name)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{user_folder}/{name}_{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(f"Image {count} captured.")

        cv2.imshow('Registering Face', frame)

        # Break after capturing 10 images or pressing 'q'
        if count >= 10 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Registration complete.")

name = input("Enter the user's name: ")
register_user(name)