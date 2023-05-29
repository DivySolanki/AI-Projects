import cv2

# Training the classifier using harcascade dataset
# Download it from here: https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier("Paste the file Path Here")

# capturing frames of video stream from webcam
webcam = cv2.VideoCapture(0)

# looping through the input received from the cam
while True:

    # destructuring the output and storing it into variables successfull_img_read is only just a boolean which signifies the success of read() and the img contains the list of image coordinates
    successfull_img_read, img = webcam.read()

    #  conversion of the img into grayscale as haarcascade is designed to work on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # capturing the face coordinates of different scale using detectMultiScale 
    face_coordinates = trained_face_data.detectMultiScale(gray)

    # looping through them and creating rectangles around it
    for x, y, w, h in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Displaying the image
    cv2.imshow("Face Detector", img)
    key = cv2.waitKey(1)

    # Press Q to quit out of WebCam
    if key == 81 or key == 113:
        break
