import numpy as np

class Kalman:
    def __init__(self, dt, num_itr, parm_RW, parm_OB):
        self.dt = dt
        self.num_itr = num_itr
        self.parm_RW = parm_RW
        self.parm_OB = parm_OB

    def simulate_world(self):
        dt, num_itr = self.dt, self.num_itr
        parm_RW = self.parm_RW

        x_real = np.zeros((2, num_itr + 1))
        u = np.zeros(num_itr + 1)
        z = np.zeros(num_itr + 1)

        Ak = np.array([[1, dt],
                       [0, 1]])
        Bk = np.array([[dt**2 / 2],
                       [dt]])
        Ck = np.array([[1, 0]])

        x_real[:, 0] = np.array([parm_RW['p_0'], parm_RW['v_0']])
        u_cmd = parm_RW['u_cmd']
        sigma_acc = parm_RW['sigma_acc']
        sigma_range = parm_RW['sigma_range']

        for i in range(1, num_itr + 1):
            x_real[:, i] = Ak @ x_real[:, i - 1] + (Bk * u_cmd).flatten()
            z[i] = Ck @ x_real[:, i] + sigma_range * np.random.randn()
            u[i] = u_cmd + sigma_acc * np.random.randn()

        return x_real, u, z

    def run_filter(self, u, z):
        dt, num_itr = self.dt, self.num_itr
        parm_OB = self.parm_OB

        mu_bar = np.zeros((2, num_itr + 1))
        mu = np.zeros((2, num_itr + 1))
        cov_bar = np.zeros((2, 2, num_itr + 1))
        cov = np.zeros((2, 2, num_itr + 1))
        z_bar = np.zeros(num_itr + 1)
        inn = np.zeros(num_itr + 1)

        Ak = np.array([[1, dt],
                       [0, 1]])
        Bk = np.array([[dt**2 / 2],
                       [dt]])
        Ck = np.array([[1, 0]])

        Rk = parm_OB['sigma_acc']**2 * np.array([[dt**4 / 4, dt**3 / 2],
                                                 [dt**3 / 2, dt**2]])
        Qk = parm_OB['sigma_range']**2

        mu[:, 0] = parm_OB['mu_0']
        cov[:, :, 0] = parm_OB['cov_0']

        for i in range(1, num_itr + 1):
            mu_bar[:, i] = Ak @ mu[:, i - 1] + (Bk * u[i]).flatten()
            cov_bar[:, :, i] = Ak @ cov[:, :, i - 1] @ Ak.T + Rk

            Kk = cov_bar[:, :, i] @ Ck.T / (Ck @ cov_bar[:, :, i] @ Ck.T + Qk)
            z_bar[i] = Ck @ mu_bar[:, i]
            inn[i] = z[i] - z_bar[i]
            mu[:, i] = mu_bar[:, i] + (Kk.flatten() * inn[i])
            cov[:, :, i] = (np.eye(2) - Kk @ Ck) @ cov_bar[:, :, i]

        return mu, cov, z_bar, inn

    def run(self):
        x_real, u, z = self.simulate_world()
        mu, cov, z_bar, inn = self.run_filter(u, z)
        return mu, cov, z_bar, inn, z, x_real
