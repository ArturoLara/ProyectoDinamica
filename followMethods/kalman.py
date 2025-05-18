import numpy as np


class Kalman:
    def __init__(self, dt, process_noise=0.1, measurement_noise=10):

        self.state = np.zeros(4)
        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1000  # Alta incertidumbre inicial

    def predict(self):

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:2]

    def update(self, measurement):

        if measurement is None:
            return

        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
