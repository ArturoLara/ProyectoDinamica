import numpy as np
import cv2
import argparse


class Particle:
    def __init__(self, roi_size, img, parent=None):
        self.world_size = img.shape[::-1]  # (width, height)

        if parent is None:
            # Inicialización aleatoria uniforme
            self.x = np.random.uniform(0, self.world_size[0])
            self.y = np.random.uniform(0, self.world_size[1])
            self.vx = 0
            self.vy = 0
        else:
            # Difusión con distribución normal alrededor de la posición padre
            self.x = parent.x + np.random.normal(0, roi_size / 2)
            self.y = parent.y + np.random.normal(0, roi_size / 2)
            self.vx = parent.vx + np.random.normal(0, 1)
            self.vy = parent.vy + np.random.normal(0, 1)

        # Asegurar límites de la imagen
        self.x = np.clip(self.x, 0, self.world_size[0] - 1)
        self.y = np.clip(self.y, 0, self.world_size[1] - 1)

        self.ax = 0
        self.ay = 0
        self.img = img
        self.roi_size = roi_size

    # En la clase Particle
    def move(self, use_vel=True, use_accel=False):
        dt = 1
        noise_pos = 2  # Aumentar ruido posicional
        noise_vel = 1  # Aumentar ruido en velocidad

        if use_vel:
            self.x += self.vx * dt + np.random.normal(0, noise_pos)
            self.y += self.vy * dt + np.random.normal(0, noise_pos)

        if use_accel:
            self.vx += np.random.normal(0, noise_vel)
            self.vy += np.random.normal(0, noise_vel)

        # Limitar velocidad máxima
        max_speed = 20
        self.vx = np.clip(self.vx, -max_speed, max_speed)
        self.vy = np.clip(self.vy, -max_speed, max_speed)

    def sense(self):
        x, y = int(self.x), int(self.y)
        half = self.roi_size // 2

        # Definir región de interés (ROI)
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(self.world_size[0], x + half)
        y2 = min(self.world_size[1], y + half)

        # Manejar casos donde el ROI está fuera de la imagen
        if x1 >= x2 or y1 >= y2:
            return 0.0

        roi = self.img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Calcular porcentaje de blancos en el ROI
        white = np.sum(roi == 255)
        total = roi.size
        return white / total if total > 0 else 0.0


class PF:
    def __init__(self, roi_size, img, n_particles=100, video=None,
                 use_vel=True, use_accel=False, resample="pr"):
        self.video = video
        self.use_vel = use_vel
        self.use_accel = use_accel
        self.resample_mode = resample
        self.n_particles = n_particles
        self.roi_size = roi_size
        self.smooth_factor = 0.2
        self.last_estimate = None

        # Inicializar con primera imagen
        self.img = img if video is None else video[0]
        self.particles = [Particle(roi_size, self.img)
                          for _ in range(n_particles)]
        self.weights = np.ones(n_particles) / n_particles
        self.stored_particles = [self.particles]

    def get_smoothed_estimate(self):
        current = self.get_estimate()
        if self.last_estimate is None:
            self.last_estimate = current
            return current

        # Filtro de suavizado exponencial
        smoothed = (self.smooth_factor * current +
                    (1 - self.smooth_factor) * self.last_estimate)
        self.last_estimate = smoothed
        return smoothed

    def _systematic_resample(self):
        indices = np.zeros(self.n_particles, dtype=int)
        cumulative = np.cumsum(self.weights)
        step = cumulative[-1] / self.n_particles
        u = np.random.uniform(0, step)

        i = 0
        for j in range(self.n_particles):
            while u > cumulative[i]:
                i += 1
            indices[j] = i
            u += step

        return indices

    def _resample(self):
        indices = np.zeros(self.n_particles, dtype=int)
        cumulative = np.cumsum(self.weights)
        step = cumulative[-1] / self.n_particles
        u = np.random.uniform(0, step)

        i = 0
        for j in range(self.n_particles):
            while u > cumulative[i]:
                i += 1
            indices[j] = i
            u += step

        new_particles = []
        for i in indices:
            p = self.particles[i]
            new_p = Particle(self.roi_size, self.img, p)

            # Suavizar transición entre posiciones
            new_p.x += np.random.normal(0, 0.5)
            new_p.y += np.random.normal(0, 0.5)

            new_particles.append(new_p)

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def run(self, steps=1):
        for _ in range(steps):
            # 1. Movimiento de partículas
            for p in self.particles:
                p.move(self.use_vel, self.use_accel)

            # 2. Calcular pesos
            weights = np.array([p.sense() for p in self.particles])
            weights += 1e-300  # Evitar división por cero
            self.weights = weights / weights.sum()

            # 3. Resampleo
            if 1.0 / np.sum(self.weights ** 2) < self.n_particles / 2.0:
                self._resample()

            self.stored_particles.append(self.particles.copy())

    def get_estimate(self):
        # Usar promedio ponderado en lugar de mejor partícula
        x = sum(p.x * w for p, w in zip(self.particles, self.weights))
        y = sum(p.y * w for p, w in zip(self.particles, self.weights))
        return np.array([x, y])

