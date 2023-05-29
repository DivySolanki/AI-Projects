import cv2

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    successfull_img_read, img = webcam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray)

    for x, y, w, h in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detector", img)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

    # webcam.release()
    # print("Code completed")
