import cv2
import numpy as np
import scipy.stats

def background_subtraction_mean(imgs, threshold_value=40, window_size=5):
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    result = []

    for i in range(len(gray_imgs)):
        # Determina el inicio de la ventana, sin salirse del índice 0
        start = max(0, i - window_size + 1)
        stop = min(len(imgs), start + window_size)
        # Toma las imágenes de la ventana
        window = gray_imgs[start:stop]
        # Calcula la media del fondo con las imágenes de la ventana
        avg_background = np.mean(window, axis=0).astype(np.uint8)

        diff = cv2.absdiff(gray_imgs[i], avg_background)
        _, thresholded = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        result.append(thresholded)

    return result

def background_subtraction_exponential_moving_average(imgs, alpha=0.05, threshold_value=40):
    gray_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs])
    background = gray_imgs[0]
    result = []

    for frame in gray_imgs[1:]:
        # Calcula la media del fondo con las imágenes de la ventana
        avg_background = (alpha * frame + (1 - alpha) * background).astype(np.uint8)

        diff = cv2.absdiff(frame, avg_background)
        _, thresholded = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        result.append(thresholded)

    return result

def background_subtraction_exponential_moving_average_consider_bg(imgs, alpha=0.05, threshold_value=40, window_size=5):
    gray_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs])
    background = gray_imgs[0]
    result = []

    for frame in gray_imgs[1:]:
        # Calcula la media del fondo con las imágenes de la ventana
        diff = cv2.absdiff(frame, background)
        _, thresholded = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        background = np.where(thresholded == 0, (alpha * frame + (1 - alpha) * background).astype(np.uint8), background)
        result.append(thresholded)

    return result

def background_subtraction_gaussian_moving_average(imgs, alpha=0.05, k=2.5):
    gray_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]) 
    mean = gray_imgs[0].astype(np.float32)
    variance = np.ones_like(mean, dtype=np.float32) * 10  # Inicialización de la varianza
    result = []
    
    for frame in gray_imgs[1:]:
        mean = alpha * frame + (1 - alpha) * mean
        
        # Actualización de la varianza
        variance = alpha * (frame - mean) ** 2 + (1 - alpha) * variance
        std_dev = np.sqrt(variance)
        
        # Detección de primer plano
        diff = np.abs(frame - mean)
        foreground_mask = (diff > k * std_dev).astype(np.uint8) * 255
        result.append(foreground_mask)
    return result