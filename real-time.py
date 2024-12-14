import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.models import Sequential

# Load the trained model
json_file = open(r"C:\Users\Laptronics.co\PycharmProjects\PythonProject2\BackgroundRemoval\facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()

# Deserialize the model with custom_objects
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})

# Load weights into the model
model.load_weights(r"C:\Users\Laptronics.co\PycharmProjects\PythonProject2\BackgroundRemoval\facialemotionmodel.h5")
print("Model loaded successfully")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open webcam
webcam = cv2.VideoCapture(0)

# Check if webcam is successfully opened
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Capture frame-by-frame
    ret, im = webcam.read()

    # If the frame is empty or reading failed, skip this loop
    if not ret:
        print("Error: Failed to capture image.")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Use gray image for face detection

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Display the resulting frame
        cv2.imshow("Output", im)

        # Wait for key press, exit if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII for ESC key
            break

    except cv2.error:
        print("Error during face detection or prediction.")
        pass

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()