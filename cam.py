import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)
while True:
    #Capture frame-by-frame
    __, frame = cap.read()

    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']

            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,0,255),
                          2)

        (x, y) = (bounding_box[0], bounding_box[1])
        (w, h) = (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])

        for i in range(len(result)):
            text = "{:.4f}".format(result[i]['confidence'])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0) , 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()



