import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Création de la figure et de l'axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Données pour la surface
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(X) * np.cos(Y)

# Initialisation de la surface
surface = [ax.plot_surface(X, Y, Z, cmap='viridis')]

# Fonction d'initialisation de l'animation
def init():
    return surface

# Fonction d'animation
def animate(i):
    # Modification de la surface pour créer une animation
    Z = np.sin(X + 0.1 * i) * np.cos(Y + 0.1 * i)
    ax.clear()  # Effacer l'axe pour éviter les superpositions
    surface[0] = ax.plot_surface(X, Y, Z, cmap='viridis')
    return surface

# Création de l'animation
ani = FuncAnimation(fig, animate, frames=100, init_func=init, interval=50, blit=False)

plt.show()
