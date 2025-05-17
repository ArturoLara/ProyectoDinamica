# python3 practica.py --input_path="./SecuenciaPelota/" --filter="cf" --use_vel="true" --use_accel=true --particle_sel='nr'
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class Particle:
  def __init__(self, roi_size, img, parent_particle=None):
    if parent_particle == None: # esto se correspondería a la inicialización
      self.x = np.random.rand()*img.shape[0]
      self.y = np.random.rand()*img.shape[1]
      self.vx = 0
      self.vy = 0
      self.ax = 0
      self.ay = 0
      self.move_random = True
    else: # esto se correspondería a la difusión
        self.x = parent_particle.x + (np.random.randint(0, roi_size) - roi_size/2)
        self.y = parent_particle.y + (np.random.randint(0, roi_size) - roi_size/2)
        # no hace falta que sea random la vel y accel ya que la velocidad del movimiento de la particula
        # variará en funcion de la posicion actual que ya es aleatoria y la de la posición anterior
        dt = 0.005
        self.vx = (self.x - parent_particle.x) // dt
        self.vy = (self.y - parent_particle.y) // dt
        self.ax = (self.vx - parent_particle.vx) // dt
        self.ay = (self.vy - parent_particle.vy) // dt
        self.move_random = False

    self.img = img
    self.world_size = img.shape
    self.roi_size = roi_size
    # Asegurarse de que self.x y self.y estén dentro de los límites de la imagen
    self.x = max(0, min(self.x, img.shape[0]))
    self.y = max(0, min(self.y, img.shape[1]))
    self.traj = [self.get_loc()]

  def move(self, use_vel, use_accel):
    # si se mueve aleatoriamente o no uso ni la velocidad ni la aceleración, no hace falta que mueva la párticula, ya se ha movido aleatoríamente al resamplearse
    if self.move_random or (not use_vel and not use_accel):
      self.traj.append(self.get_loc())
    
    else:
      # dt hacemos que sea 2 fotogramas
      dt = 0.005

      if use_vel == True:
        self.x = self.x + self.vx*dt
        self.y = self.y + self.vy*dt
      if use_accel:
        self.x = self.x + 0.5* self.ax * dt **2
        self.y = self.y + 0.5 * self.ay * dt **2
      
      #Asegurarse de que self.x y self.y estén dentro de los límites de la imagen
      self.x = max(0, min(self.x, self.world_size[0] - 1))
      self.y = max(0, min(self.y, self.world_size[1] - 1))

      self.traj.append(self.get_loc())

  def sense(self):
    half_size = self.roi_size // 2

    # Definir los límites del roi asegurándose de no salir de la imagen
    x1, y1 = int(max(0, self.x - half_size)), int(max(0, self.y - half_size))
    x2, y2 = int(min(self.world_size[0], self.x + half_size)), int(min(self.world_size[1], self.y + half_size))
    
    roi = self.img[int(y1):int(y2), int(x1):int(x2)]

    white_pixels = np.sum(roi == 255)
    height, width = roi.shape[:2]
    # divido entre los pixeles totales para que los rois pequeños no tengan desventaja por si estuvieran en los bordes
    return white_pixels / (height * width)

  def get_loc(self):
    return np.array([self.x, self.y])

  def plot_traj(self, ax, init_col='ro', arrow_col='black', stop=None):
    if stop == None or stop < 1 or stop > len(self.traj):
      stop = len(self.traj)
    for i in range(1, stop):
      delta_loc = self.traj[i] - self.traj[i - 1]
      ax.arrow(self.traj[i-1][0], self.traj[i-1][1], delta_loc[0], delta_loc[1], head_width=0.15, color=arrow_col)
      
    ax.plot(self.traj[stop-1][0], self.traj[stop-1][1], init_col, markerfacecolor='None')
    ax.axis([0, self.world_size[0], 0, self.world_size[1]])
    ax.set_aspect('equal')

class PF:
  # img = imagen donde la párticula va a buscar pixeles blancos (si no se usa el parámetro video)
  # vide = array de imágenes donde las particulas van a buscar pixels blancos
  # use_vel y use_accel = si se usa la posición y la velocidad para el modelado del movimiento de la partícula
  # resample = método de resampleo de las particulas -> nr: ruleta normal; pr: metodo low-variance
  def __init__(self, roi_size, img, n_particles=10, video=None, use_vel=True, use_accel=True, resample="nr"):
    self.video = video
    self.use_vel = use_vel
    self.use_accel = use_accel
    self.resample_mode = resample
    
    if self.video == None:
      self.img = img
    else:
      self.img = self.video[0]

    self.n_particles = n_particles

    self.roi_size=roi_size
    self._recreate_particles()
    self.stored_particles = [self.particles]

  def _recreate_particles(self):
    self.particles = [Particle(self.roi_size, self.img) for i in range(self.n_particles)]

  def _move(self):
    for particle in self.particles:
      particle.move(self.use_vel, self.use_accel)

  def _resample_lowvar(self, weights):
    """Resamplear segun el peso de la particula y low-variance"""
    # Obtener índices aleatorios de n_partículas
    index = np.random.randint(0, self.n_particles, size=self.n_particles)
    # Obtener pesos en índices aleatorios
    idxed_weights = weights[index]
    # Inicializar beta a 2 veces el peso máximo
    beta = 2.0*np.max(weights)*np.random.rand(self.n_particles)
    # Hacer arr bool para encontrar qué pesos indexados son menores que beta
    are_less = idxed_weights < beta
    # Si aún quedan ponderaciones indexadas inferiores a beta
    while np.any(are_less):
      # Disminuir betas que son más grandes que pesos.
      beta[are_less] -= idxed_weights[are_less]
      # Índices de incremento circular en los que los pesos eran inferiores a beta.
      index[are_less] = (index[are_less] + 1) % self.n_particles
      idxed_weights = weights[index]
      # Vuelva a comprobar si los pesos siguen siendo inferiores a beta
      are_less = idxed_weights < beta
    # index debe tener índices de partículas elegidas del proceso de remuestreo. 
    particles = [Particle(roi_size=self.roi_size, img=self.img, parent_particle=self.particles[index[i]]) \
                 for i in range(self.n_particles)]
    # Devuelve las copias de partículas remuestreadas.
    return particles
  
  def _resample_roulete(self, weights):
    """Resample according to importance weights"""
    indices_seleccionados = []
    while len(indices_seleccionados) < len(weights):
      # Obtener los índices que ordenan los weights de mayor a menor
      indices_ordenados_desc = np.argsort(weights)[::-1]
      
      # Generar un número aleatorio entre 1 y la cantidad de weights
      num_seleccionados = np.random.randint(1, len(weights))
      
      # Obtener los índices ordenados de menor a mayor para los primeros num_seleccionados weights
      seleccionados = np.sort(indices_ordenados_desc[:num_seleccionados])
      
      # Agregar los nuevos índices asegurando que el tamaño final sea el adecuado
      indices_seleccionados.extend(seleccionados.tolist())
      indices_seleccionados = indices_seleccionados[:len(weights)]
    
    # index debe tener índices de partículas elegidas del proceso de remuestreo. 
    particles = [Particle(roi_size=self.roi_size, img=self.img, parent_particle=self.particles[indices_seleccionados[i]]) \
                  for i in range(self.n_particles)]
    # Devuelve las copias de partículas remuestreadas.
    return particles

  def run(self, steps=10):
    for step in range(steps):
      if self.video != None:
        self.img = self.video[step]
      # mueve las partículas como corresponda
      self._move()

      # Establece el peso de importancia de la partícula según cual haya contado más pixels blancos.
      weights = np.array([particle.sense() for particle in self.particles])

      # Remuestreo de partículas en función de su peso.
      if self.resample_mode == "nr":
        self.particles = self._resample_roulete(weights)
      elif self.resample_mode == "pr":
        self.particles = self._resample_lowvar(weights)
      else:
        raise argparse.ArgumentTypeError('only normal_roulette(nr) or pondered_roulette_(pr) available')
         
      self.stored_particles.append(self.particles)

  def plot_results(self, mcl_steps, n_plot_cols, n_plot_rows):
    fig, ax = plt.subplots(n_plot_rows, n_plot_cols,figsize=(n_plot_cols*5, n_plot_rows*5))

    for i in range(n_plot_rows):
      for j in range(n_plot_cols):
        idx = j + i*n_plot_cols
        if idx == mcl_steps:
          break

      # Dibujar la imagen de fondo en el subplot
        if self.video == None:
          ax[i, j].imshow(self.img) 
        else:
          ax[i, j].imshow(self.video[idx])  
      
        particles = self.stored_particles[idx]
        for particle in particles:
          particle.plot_traj(ax[i,j], init_col='mo', arrow_col='grey')
        ax[i,j].invert_yaxis() # como calculamos con opencv pero graficamos con matlab hay que invertir el eje y de la representación
    fig.tight_layout()
    plt.show()