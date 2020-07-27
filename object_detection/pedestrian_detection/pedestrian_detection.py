import os
import cv2
import imutils

hgd = cv2.HOGDescriptor()
hgd.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for filename in os.listdir('pedestrian_images'):
    print(filename)
    image = cv2.imread(os.path.join('pedestrian_images', filename))
    if image is not None:  # and 'pedestrian6' in filename:
        image = imutils.resize(image, width=min(1000, image.shape[1]))
        # orig = image.copy()
        (regions, weights) = hgd.detectMultiScale(image, winStride=(2, 2), padding=(4, 4), scale=1.05)
        print(regions)
        if len(regions):
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(0)
