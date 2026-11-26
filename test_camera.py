
import cv2
#commit
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Webcam not detected!")
    exit()

print("Webcam started!")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to capture frame")
            break

        # Correct mirrored webcam image
        frame = cv2.flip(frame, 1)

        cv2.imshow("Original Webcam View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
