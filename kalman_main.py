from followMethods import Kalman
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
dt = 0.1
num_itr = 100

parm_RW = {
    'p_0': 0.0,
    'v_0': 0.0,
    'u_cmd': 1.0,
    'sigma_acc': 0.2,
    'sigma_range': 1.0
}

parm_OB = {
    'mu_0': np.array([0.0, 0.0]),
    'cov_0': np.eye(2),
    'sigma_acc': 0.2,
    'sigma_range': 1.0
}

kf =  Kalman(dt, num_itr, parm_RW, parm_OB)
mu, cov, z_bar, inn, z, x_real = kf.run()

# Visualización
time = np.linspace(0, num_itr * dt, num_itr + 1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x_real[0], label='Posición real')
plt.plot(time, z, 'r.', label='Mediciones')
plt.plot(time, mu[0], label='Estimación (KF)')
plt.ylabel('Posición')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, x_real[1], label='Velocidad real')
plt.plot(time, mu[1], label='Estimación (KF)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
