import cv2
import numpy as np
import matplotlib.pyplot as plt

from followMethods import Kalman, PF


def detectar_objeto(img_binaria):

    contours, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)

    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.array([cx, cy])
    return None

def procesar_secuencia(video, bs, dt, use_vel, use_accel, resample, particulas):

    result = []

    # Inicializar filtros
    kalman = Kalman(dt=dt)

    pf = PF(roi_size=20, img=bs[0], n_particles=particulas,
            video=bs, use_vel=use_vel, use_accel=use_accel,
            resample=resample)

    # Variables para seguimiento
    trayectorias = {
        'mediciones': [],
        'kalman': [],
        'particulas': []
    }

    for idx, frame in enumerate(bs):
        # Detección del objeto
        medicion = detectar_objeto(frame)

        # Actualizar Kalman
        if idx == 0 and medicion is not None:
            kalman.state[:2] = medicion
            estimacion_kf = kalman.state[:2]
        else:
            estimacion_kf = kalman.predict()
            if medicion is not None:
                kalman.update(medicion)

        # Actualizar Particle Filter
        if idx > 0:
            pf.run(steps=1)  # Ejecutar un paso del filtro de partículas
            mejor_particula = max(pf.particles, key=lambda p: p.sense())
            estimacion_pf = mejor_particula.get_loc()
        else:
            estimacion_pf = medicion if medicion is not None else np.zeros(2)

        # Almacenar resultados
        trayectorias['mediciones'].append(medicion.copy() if medicion is not None else np.array([np.nan, np.nan]))
        trayectorias['kalman'].append(estimacion_kf.copy())
        trayectorias['particulas'].append(estimacion_pf.copy())

        # Visualización
        img_color = video[idx]

        # Dibujar elementos
        if medicion is not None:
            cv2.circle(img_color, tuple(medicion.astype(int)), 5, (0, 0, 255), -1)  # Medicion (rojo)

        cv2.circle(img_color, tuple(estimacion_kf.astype(int)), 5, (0, 255, 0), -1)  # Kalman (verde)
        cv2.circle(img_color, tuple(estimacion_pf.astype(int)), 5, (255, 0, 255), -1)  # Partículas (magenta)

        # Dibujar trayectorias
        for i in range(1, len(trayectorias['kalman'])):
            cv2.line(img_color,
                     tuple(trayectorias['kalman'][i - 1].astype(int)),
                     tuple(trayectorias['kalman'][i].astype(int)),
                     (0, 255, 0), 2)  # Trayectoria Kalman

            cv2.line(img_color,
                     tuple(trayectorias['particulas'][i - 1].astype(int)),
                     tuple(trayectorias['particulas'][i].astype(int)),
                     (255, 0, 255), 2)  # Trayectoria Partículas

            if trayectorias['mediciones'][i - 1] is not None:
                cv2.line(img_color,
                         tuple(trayectorias['mediciones'][i - 1].astype(int)),
                         tuple(trayectorias['mediciones'][i].astype(int)),
                         (0, 0, 255), 1)  # Trayectoria Mediciones

        # Añadir leyenda
        cv2.putText(img_color, "Medicion (Rojo)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_color, "Kalman (Verde)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_color, "Particulas (Magenta)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        result.append(img_color)

    return result, trayectorias



def graficar_resultados(trayectorias, output):

    plt.figure(figsize=(15, 8))

    # Extraer coordenadas
    med_x = [m[0] for m in trayectorias['mediciones'] if not np.isnan(m[0])]
    med_y = [m[1] for m in trayectorias['mediciones'] if not np.isnan(m[1])]

    kalman_x = [k[0] for k in trayectorias['kalman']]
    kalman_y = [k[1] for k in trayectorias['kalman']]

    pf_x = [p[0] for p in trayectorias['particulas']]
    pf_y = [p[1] for p in trayectorias['particulas']]

    # Graficar trayectorias
    plt.plot(med_x, med_y, 'ro', label='Mediciones', markersize=4)
    plt.plot(kalman_x, kalman_y, 'g-', linewidth=2, label='Filtro de Kalman')
    plt.plot(pf_x, pf_y, 'm--', linewidth=2, label='Filtro de Partículas')

    plt.title('Comparativa de Trayectorias')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output)

def extraer_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames


def guardar_video(frames, output_path, fps=30, codec='mp4v'):

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    # Liberar recursos
    video.release()

if __name__ == "__main__":

    path_video = "dataset/video/Walking.54138969.mp4"
    path_bs = "dataset/bs/Walking.54138969.mp4"
    path_output = "dataset/output/Walking.54138969"

    video = extraer_frames(path_video)
    frames = extraer_frames(path_bs)

    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    result, trayectorias = procesar_secuencia(video, frames, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=100)

    guardar_video(result, path_output + ".mp4", fps=30)
    graficar_resultados(trayectorias, path_output + ".png")
