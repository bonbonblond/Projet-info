import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Création de la figure et de l'axe
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r.', animated=True)

# Fonction d'initialisation de l'axe
def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.5)
    return ln,

# Fonction de mise à jour du graphique à chaque frame
def update(frame):
    x = np.random.normal(0, 1, 1000)  # Données aléatoires selon une loi normale
    y, _ = np.histogram(x, bins=50, density=True)
    x = (_[1:] + _[:-1]) / 2
    ln.set_data(x, y)
    return ln,

# Création de l'animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 200), init_func=init, blit=True)

plt.show()
