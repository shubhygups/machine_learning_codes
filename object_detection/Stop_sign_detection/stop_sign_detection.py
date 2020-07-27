import os
import cv2
from matplotlib import pyplot as plt

for filename in os.listdir('stop_sign_images'):
    image = cv2.imread(os.path.join('stop_sign_images', filename))
    if image is not None:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stop_data = cv2.CascadeClassifier('stop_data.xml')
        region = stop_data.detectMultiScale(image_gray, minSize=(20, 20))
        if len(region):
            for (x, y, width, height) in region:
                # print("{}-{}-{}-{}".format(x, y, x + height, y + width))
                cv2.rectangle(image_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)

        plt.subplot(1, 1, 1)
        plt.imshow(image_rgb)
        plt.show()
