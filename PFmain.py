import cv2
import os
import numpy as np
from natsort import natsorted

from followMethods import PF
from subtractionMethods import *


def sort_imgs_path(path):
    imgs_name = os.listdir(path)
    imgs_name = natsorted(imgs_name)  # python sorts strings= [1 3 10 2] as [1 10 2 3] by default so I need natsorted
    imgs_sorted = [os.path.join(path, img) for img in imgs_name]
    return imgs_sorted

def extraer_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video o error de lectura
        frames.append(frame)

    cap.release()
    return frames

if __name__ == "__main__":

    input_path = "dataset/video/Walking.54138969.mp4"
    readed_imgs = extraer_frames(input_path)

    processed_imgs = []

    processed_imgs = background_subtraction_mean(readed_imgs, threshold_value=40)

    # Particle Filter variables
    num_particles = 100
    particle_roi_size = 16

    use_vel = True
    use_accel = True
    particle_sel = "nr"  # nr: normal roulette; pr: low variance

    # Plotting variables
    mcl_steps = len(processed_imgs)
    n_plot_cols = 10
    n_plot_rows = int(np.ceil(mcl_steps / n_plot_cols))  # Solo puedo usar un int para el numero de filas.

    mcl = PF(particle_roi_size, processed_imgs[20], n_particles=num_particles, video=processed_imgs,
             use_vel=use_vel, use_accel=use_accel, resample=particle_sel)

    mcl.run(steps=mcl_steps)

    mcl.plot_results(mcl_steps, n_plot_cols, n_plot_rows)