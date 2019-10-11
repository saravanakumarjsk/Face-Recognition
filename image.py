import cv2
from mtcnn.mtcnn import MTCNN


# adjust the min_face_size for more fine tuning
detector = MTCNN(min_face_size=10)

# pass in the test image
image = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)

font = cv2.FONT_HERSHEY_PLAIN


# Result is an array with all the bounding boxes detected.
for i in range(len(result)):
    bounding_box = result[i]['box']
    confidence = result[i]['confidence']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,0,255),
                  2)

    (x, y) = (bounding_box[0], bounding_box[1])
    (w, h) = (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])



    text = "{:.4f}".format(result[i]['confidence'])
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0) , 2)

    cv2.imshow("detector", image)
cv2.waitKey(0)

# print(result, '\n')
print(len(result))

