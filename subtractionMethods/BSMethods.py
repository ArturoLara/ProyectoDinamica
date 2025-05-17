import cv2
import numpy as np

def background_subtraction_mean(imgs, threshold_value=20):
    gray_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs])
    # las 6 primeras imagenes no tienen mucha parte de la pelota, si uso todas la img media tiene una pelota en el centro.
    avg_background = np.mean(gray_imgs[:6], axis=0).astype(np.uint8)
    result = []

    for gray_img in gray_imgs:
        diff = cv2.absdiff(gray_img, avg_background)
        _, thresholded = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        result.append(thresholded)

    return result