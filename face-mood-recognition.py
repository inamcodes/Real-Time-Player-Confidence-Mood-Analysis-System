import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier = load_model('model.h5')


def predict_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        face = face.astype('float') / 255.0
        face = np.expand_dims(img_to_array(face), axis=0)

        prediction = classifier.predict(face)[0]
        label = emotion_labels[prediction.argmax()]
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def start_webcam():
    print("[Action] Starting Webcam... Press 'q' inside the webcam window to close it.")
    cap = cv2.VideoCapture(0)
    cap.set(10, 10)

    while True:
        sucess, frame = cap.read()
        if not sucess:
            break

        processed_frame = predict_emotions(frame)
        cv2.imshow("Webcam - press 'q' to close", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Webcam - press 'q' to close")
    cv2.imshow("Main Panel - Press 'q' to quit", bg)


def upload_image():
    print("[Action] Opening Image Upload Dialog...")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select Image for Emotion Recognition",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        print(f"Loading: {file_path}")
        image = cv2.imread(file_path)

        if image is not None:

            max_dimension = 640
            height, width = image.shape[:2]

            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                print(
                    f"Resizing from {width}x{height} to {new_width}x{new_height} to fit boundaries.")
                image = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            processed_image = predict_emotions(image)

            cv2.imshow('Uploaded Image Results', processed_image)
            print("Displaying results. Press any key to close this view.")
            cv2.waitKey(0)
            cv2.destroyWindow('Uploaded Image Results')
        else:
            print("Error: Selected file could not be read as an image.")
    else:
        print("Upload cancelled by user.")


# --- Mouse Callback Event Handler ---
def handle_clicks(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_webcam_start[0] <= x <= btn_webcam_end[0] and btn_webcam_start[1] <= y <= btn_webcam_end[1]:
            start_webcam()

        elif btn_upload_start[0] <= x <= btn_upload_end[0] and btn_upload_start[1] <= y <= btn_upload_end[1]:
            upload_image()


bg = np.zeros((480, 640, 3), dtype=np.uint8)

btn_webcam_start = (120, 200)
btn_webcam_end = (290, 260)

btn_upload_start = (350, 200)
btn_upload_end = (520, 260)

cv2.rectangle(bg, btn_webcam_start, btn_webcam_end, (0, 200, 0), -1)
cv2.putText(bg, "Webcam", (155, 238),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.rectangle(bg, btn_upload_start, btn_upload_end, (200, 0, 0), -1)
cv2.putText(bg, "Upload Image", (365, 238),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.namedWindow("Main Panel - Press 'q' to quit")
cv2.setMouseCallback("Main Panel - Press 'q' to quit", handle_clicks)

print("Main Panel active. Choose an option inside the window interface.")

while True:
    cv2.imshow("Main Panel - Press 'q' to quit", bg)

    if cv2.waitKey(1) & 0xFF != 255:
        break

cv2.destroyAllWindows()
