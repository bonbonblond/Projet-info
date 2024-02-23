import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis object
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)

# Set axis limits
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1, 1)

# Initialization function: plot the background of each frame
def init():
    ln.set_data([], [])
    return ln,

# Animation function which updates the plot
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# Generate frames
frames = np.linspace(0, 2*np.pi, 128)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

plt.show()
