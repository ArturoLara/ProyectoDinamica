import cv2
import numpy as np
import matplotlib.pyplot as plt

from followMethods import Kalman, PF, Swarm
from subtractionMethods import *
from ultralypip3tics import YOLO


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


def procesar_secuencia(video, bs, gt, dt, use_vel, use_accel, resample, particulas):
    result = []

    # Inicializar filtros
    kalman = Kalman(dt=dt, process_noise=0.1, measurement_noise=10)

    pf = PF(
        roi_size=40,
        img=bs[0],
        n_particles=particulas,
        video=bs,  # Pasar toda la secuencia binaria
        use_vel=use_vel,
        use_accel=use_accel,
        resample=resample
    )

    swarm = Swarm(
        roi_size=40,
        img=bs[0],
        n_particles=particulas,
        video=bs,  # Pasar toda la secuencia binaria
        use_vel=use_vel,
        use_accel=use_accel,
        resample=resample
    )

    # Cargar modelo YOLOv8 preentrenado
    model = YOLO('yolov8n.pt')  # Puedes usar yolov8n.pt, yolov8s.pt, etc.

    # Variables para seguimiento
    trayectorias = {
        'mediciones': [],
        'kalman': [],
        'particulas': [],
        'enjambre': [],
        'yolo': [],
        'ground_truth': []
    }

    last_position_pf = None
    last_position_swarm = None

    good_contours = -1

    for idx, (frame, gt_frame) in enumerate(zip(bs, gt)):
        # 1. Detección del objeto
        medicion = detectar_objeto(frame)
        gt_pos = detectar_objeto(gt_frame)
        if medicion is None:
            continue
        good_contours+=1
        # 2. Actualizar Kalman
        if good_contours == 0 and medicion is not None:
            kalman.state[:2] = medicion
            estimacion_kf = kalman.state[:2]
        else:
            estimacion_kf = kalman.predict()
            if medicion is not None:
                kalman.update(medicion)

        # 3. Actualizar Particle Filter
        if good_contours == 0:
            if medicion is not None:
                last_position_pf = medicion.copy()
            else:
                last_position_pf = np.zeros(2)
            estimacion_pf = last_position_pf.copy()
        else:
            pf.img = frame  # Actualizar frame actual en el PF
            pf.run(steps=1)
            estimacion_pf = pf.get_smoothed_estimate()

            # Actualización adaptativa del ROI
            if np.linalg.norm(estimacion_pf - last_position_pf) > 10:
                pf.roi_size = min(60, pf.roi_size + 10)
            else:
                pf.roi_size = max(30, pf.roi_size - 5)

        last_position_pf = estimacion_pf.copy()

        # 4. Actualizar Swarm
        if good_contours == 0:
            if medicion is not None:
                last_position_swarm = medicion.copy()
            else:
                last_position_swarm = np.zeros(2)
            estimacion_swarm = last_position_swarm.copy()
        else:
            swarm.img = frame  # Actualizar frame actual en el PF
            swarm.run(steps=1)
            estimacion_swarm = swarm.get_smoothed_estimate()

            # Actualización adaptativa del ROI
            if np.linalg.norm(estimacion_swarm - last_position_swarm) > 10:
                swarm.roi_size = min(60, swarm.roi_size + 10)
            else:
                swarm.roi_size = max(30, swarm.roi_size - 5)

        last_position_swarm = estimacion_swarm.copy()

        # 4. Actualizar Yolo
        results_bbox = model(video[idx], verbose=False)
        max_conf = 0
        best_box = None
        for result_bbox in results_bbox:
            boxes = result_bbox.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if result_bbox.names[cls_id] == 'person' and conf > max_conf:
                    max_conf = conf
                    best_box = box

        # Si se encontró una persona
        estimacion_yolo = None
        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            estimacion_yolo = np.array([cx,cy])

        # 4. Almacenar resultados
        trayectorias['mediciones'].append(
            medicion.copy() if medicion is not None else np.array([np.nan, np.nan])
        )
        trayectorias['kalman'].append(estimacion_kf.copy())
        trayectorias['particulas'].append(np.array(estimacion_pf).copy())
        trayectorias['enjambre'].append(np.array(estimacion_swarm).copy())
        trayectorias['ground_truth'].append(
            gt_pos.copy() if gt_pos is not None else np.array([np.nan, np.nan])
        )
        trayectorias['yolo'].append(np.array(estimacion_yolo).copy())


        # 5. Visualización
        img_color = video[idx].copy()

        mask = (frame > 0).astype(np.uint8) * 255

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_atenuada = cv2.addWeighted(img_gray_3ch, 0.3, img_gray_3ch, 0.7, 0)
        #img_atenuada = (img_atenuada * 0.3).astype(np.uint8)
        color_zone = cv2.bitwise_and(img_color, img_color, mask=mask)
        gray_zone = cv2.bitwise_and(img_atenuada, img_atenuada, mask=cv2.bitwise_not(mask))

        img_color = cv2.add(color_zone, gray_zone)

        # Dibujar elementos
        if medicion is not None:
            cv2.circle(img_color, tuple(medicion.astype(int)), 5, (0, 0, 255), -1)  # Medicion (rojo)
        if gt_pos is not None:
            cv2.circle(img_color, tuple(gt_pos.astype(int)), 5, (0, 255, 255), -1)  # Ground truth (amarillo)
        cv2.circle(img_color, tuple(estimacion_kf.astype(int)), 5, (0, 255, 0), -1)  # Kalman (verde)
        cv2.circle(img_color, tuple(estimacion_pf.astype(int).tolist()), 5, (255, 0, 255), -1) # Partículas (magenta)
        cv2.circle(img_color, tuple(estimacion_swarm.astype(int).tolist()), 5, (255, 0, 0), -1) # Swarm (azul)
        cv2.circle(img_color, tuple(estimacion_yolo.astype(int).tolist()), 5, (125, 125, 0), -1) # Yolo (amarillo)

        # Dibujar trayectorias
        if good_contours > 0:
            # Kalman
            cv2.line(img_color,
                     tuple(trayectorias['kalman'][good_contours - 1].astype(int)),
                     tuple(estimacion_kf.astype(int)),
                     (0, 255, 0), 2)

            # Partículas
            cv2.line(img_color,
                     tuple(trayectorias['particulas'][good_contours - 1].astype(int)),
                     tuple(estimacion_pf.astype(int)),
                     (255, 0, 255), 2)
            
            cv2.line(img_color,
                     tuple(trayectorias['enjambre'][good_contours - 1].astype(int)),
                     tuple(estimacion_swarm.astype(int)),
                     (255, 0, 0), 2)

            # Ground truth
            if not np.isnan(trayectorias['ground_truth'][good_contours - 1][0]) and gt_pos is not None:
                cv2.line(img_color,
                         tuple(trayectorias['ground_truth'][good_contours - 1].astype(int)),
                         tuple(gt_pos.astype(int)),
                         (0, 255, 255), 1)

            cv2.line(img_color,
                     tuple(trayectorias['yolo'][good_contours - 1].astype(int)),
                     tuple(estimacion_yolo.astype(int)),
                     (125, 125, 0), 2)

            # Mediciones (si hay consecutivas)
            if trayectorias['mediciones'][good_contours - 1] is not None and medicion is not None:

                cv2.line(img_color,
                         tuple(trayectorias['mediciones'][good_contours - 1].astype(int)),
                         tuple(medicion.astype(int)),
                         (0, 0, 255), 1)

        # Leyenda mejorada
        cv2.putText(img_color, "Medicion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_color, "Kalman", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_color, "Particulas", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(img_color, "Enjambre", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img_color, "Ground Truth", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img_color, "Yolo", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 125, 0), 2)


        result.append(img_color)

    return result, trayectorias


def graficar_resultados(trayectorias, output_base):
    # Función auxiliar para guardar en dos archivos
    def guardar_grafica(x_data, titulo, eje_y, output):
        plt.figure(figsize=(15, 8))
        time = range(len(x_data['mediciones']))

        # Ground Truth
        gt_valid = [(t, x) for t, x in enumerate(x_data['ground_truth']) if not np.isnan(x)]
        if gt_valid:
            t_gt, x_gt = zip(*gt_valid)
            plt.plot(t_gt, x_gt, 'y-', label='Ground Truth', markersize=4)

        # Mediciones
        med_valid = [(t, x) for t, x in enumerate(x_data['mediciones']) if not np.isnan(x)]
        if med_valid:
            t_med, x_med = zip(*med_valid)
            plt.plot(t_med, x_med, 'ro', label='Mediciones', markersize=4)

        # Kalman (línea continua)
        plt.plot(time, x_data['kalman'], 'g--', linewidth=2, label='Filtro de Kalman')

        # Partículas (línea discontinua)
        plt.plot(time, x_data['particulas'], 'm--', linewidth=2, label='Filtro de Partículas')
        # enjambre (línea discontinua)
        plt.plot(time, x_data['enjambre'], 'b--', linewidth=2, label='Enjambre')
        plt.plot(time, x_data['yolo'], 'y--', linewidth=2, label='Yolo')


        plt.title(titulo)
        plt.xlabel('Tiempo (frames)')
        plt.ylabel(eje_y)
        plt.legend()
        plt.grid(True)
        plt.savefig(output)
        plt.close()

    # Preparar datos para eje X
    datos_x = {
        'mediciones': [m[0] if not np.isnan(m[0]) else np.nan for m in trayectorias['mediciones']],
        'kalman': [k[0] for k in trayectorias['kalman']],
        'particulas': [p[0] for p in trayectorias['particulas']],
        'enjambre': [p[0] for p in trayectorias['enjambre']],
        'ground_truth': [gt[0] if not np.isnan(gt[0]) else np.nan for gt in trayectorias['ground_truth']],
        'yolo': [p[0] for p in trayectorias['yolo']]
    }

    # Preparar datos para eje Y
    datos_y = {
        'mediciones': [m[1] if not np.isnan(m[1]) else np.nan for m in trayectorias['mediciones']],
        'kalman': [k[1] for k in trayectorias['kalman']],
        'particulas': [p[1] for p in trayectorias['particulas']],
        'enjambre': [p[0] for p in trayectorias['enjambre']],
        'ground_truth': [gt[1] if not np.isnan(gt[1]) else np.nan for gt in trayectorias['ground_truth']],
        'yolo': [p[1] for p in trayectorias['yolo']]
    }

    # Generar y guardar gráficas
    guardar_grafica(datos_x,
                    'Evolución Temporal de la Coordenada X',
                    'X (pixels)',
                    f"{output_base}_x.png")

    guardar_grafica(datos_y,  # Intercambiamos datos para el eje Y
                    'Evolución Temporal de la Coordenada Y',
                    'Y (pixels)',
                    f"{output_base}_y.png")


def extraer_frames(video_path, reduce_dim=4, reduce_fps=3, n_frames=20*30):
    # reduce fps takes one frame per $(reduce_fps)
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    i = 0 
    
    success, frame = cap.read()
    dim = frame[0].shape[0] // reduce_dim
    i+=1
    while success:
        if i % reduce_fps == 0:
            if reduce_dim != 1:
                frame = cv2.resize(frame, (dim, dim))
            frames.append(frame)

        if i == n_frames:
            break
        success, frame = cap.read()
        i+=1
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

    reduce_dim = 4
    video = extraer_frames(path_video, reduce_dim=reduce_dim, reduce_fps=3, n_frames=20*30)

    # # Uso de background subtraction ground truth
    frames_gt = extraer_frames(path_bs, reduce_dim=reduce_dim, reduce_fps=3, n_frames=20*30)
    frames_gt = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_gt]
    result, trayectorias = procesar_secuencia(video, frames_gt, frames_gt, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=400)

    guardar_video(result, path_output + "_groundTruth.mp4", fps=60)
    graficar_resultados(trayectorias, path_output + "_groundTruth")

    # # Uso de background subtraction por media
    frames = background_subtraction_mean(video)
    guardar_video(frames, "BS_" + path_output + "_Mean.mp4", fps=60)
    result, trayectorias = procesar_secuencia(video, frames, frames_gt, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=100)

    guardar_video(frames, path_output + "_Mean.mp4", fps=60)
    graficar_resultados(trayectorias, path_output + "_Mean")

    # Uso de background subtraction por moda
    frames = background_subtraction_exponential_moving_average(video)
    result, trayectorias = procesar_secuencia(video, frames, frames_gt, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=100)

    guardar_video(result, path_output + "_AVGMove.mp4", fps=60)
    graficar_resultados(trayectorias, path_output + "_AVGMove.png")
    
    # # Uso de background subtraction por moda
    frames = background_subtraction_exponential_moving_average_consider_bg(video)
    result, trayectorias = procesar_secuencia(video, frames, frames_gt, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=100)

    guardar_video(result, path_output + "_AVGMoveBG.mp4", fps=60)
    graficar_resultados(trayectorias, path_output + "_AVGMoveBG.png")

    # Uso de background subtraction por moda
    frames = background_subtraction_gaussian_moving_average(video)
    result, trayectorias = procesar_secuencia(video, frames, frames_gt, dt=0.1,
                                             use_vel=True, use_accel=False,
                                             resample='pr', particulas=100)

    guardar_video(result, path_output + "_Gauss.mp4", fps=60)
    graficar_resultados(trayectorias, path_output + "_Gauss.png")
    
