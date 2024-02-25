import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)
import scipy.stats as sps
from scipy import integrate


sigma_1 = [np.array([400, 30]), np.array([30, 100])]
sigma_2 = [np.array([100, 20]), np.array([20, 200])]
sigma_3 = [np.array([150, -30]), np.array([-30, 100])]

mu_1 = np.array([30, 20])
mu_2 = np.array([80, 45])
mu_3 = np.array([30, 70])


def topo(x):
    y1 = sps.multivariate_normal.pdf(x, mean=mu_1, cov=sigma_1)
    y2 = sps.multivariate_normal.pdf(x, mean=mu_2, cov=sigma_2)
    y3 = sps.multivariate_normal.pdf(x, mean=mu_3, cov=sigma_3)
    y = (y1 + y2 + y3) * (10**4)
    return y


a = 100
b = 100
f_max = (10**4) * sps.multivariate_normal.pdf(mu_3, mean=mu_3, cov=sigma_3)


def alpha_find(x, y):
    return f_max - topo([x, y])


alpha = 500 / (integrate.dblquad(alpha_find, 0, a, 0, b)[0])


def theta(x):
    return alpha * (f_max - topo(x))


def theta2(x, y):
    return theta([x, y])


# Rejection sampling for theta
def rtheta(n, a, b):
    k = 0
    pts = np.empty((0, 2))
    while k < n:
        x = a * np.random.rand(1)
        y = b * np.random.rand(1)
        u = np.random.uniform(0, alpha * f_max, 1)
        if u < theta(
            (x[0], y[0])
        ):  # remark: no need to know the normalization constant!
            pts = np.vstack((pts, [x[0], y[0]]))
            k += 1
    return pts


# Poisson simulation function with rejection sampling
def poisson2(a, b):
    Theta = integrate.dblquad(theta2, 0, a, 0, b)[0]
    N = np.random.poisson(Theta)
    return rtheta(N, a, b)


## Growth after one year
def height_2(b, d):
    return np.random.exponential(1 / (4 + b / 2 + 2 * np.exp(-d)))


## Compute distance to closest tree
def computeMinDist(X):
    n = np.shape(X)[0]
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = np.min(np.linalg.norm(np.delete(X, i, axis=0) - X[i], axis=1))
    return dist


## Definition of the probability of decimation
### h : Height of the tree
### pmax : Maximum of the probability
def probDecim(h, pmax):
    col_p = np.full((np.shape(h)[0]), pmax)
    return np.minimum(col_p, 1 / h**2)


## Intensity of the offsprings
lam = 3

## Parameter of the radius
param = 1 / 20

## Maximal probability of decimation
pmax = 0.15


## Function to simulate the forest year by year
def simForest_year_by_year(nbyears, lam, param, pmin):
    ## Simulate the position of initial trees: (x,y,time of birth)

    X = poisson2(100, 100)
    X = np.hstack((X, np.zeros((np.shape(X)[0], 1))))

    ## Heights
    # Initial heights
    h = np.zeros(np.shape(X)[0])

    # Récupération des valeurs de X et h
    stockage_valeurs_X = [X]
    stockage_valeurs_h = [h]

    for i in range(nbyears):
        ## Weather
        b = np.random.binomial(1, 2 / 5)

        ## Compute distance to closest tree
        d = computeMinDist(X)

        ## Add heights
        h_new = np.array([h[i] + height_2(b, d[i]) for i in range(len(h))])

        ## Decimate
        if b:
            decim = np.random.uniform(0, 1, np.shape(X)[0])
            living_trees = np.bool_(decim > probDecim(h_new, pmin))
            X = X[living_trees]
            h = h_new[living_trees]
        else:
            h = h_new

        if len(h) == 0:
            print("All trees have died...")
            break

        ## Find which trees have just reached 3m
        mature = np.bool_(h > 3)

        ## Spawn new trees
        for tree in X[mature]:
            N = np.random.poisson(lam)
            # rayon exponentiel
            new_trees_r = np.random.exponential(param, (N, 1))
            # angle uniforme
            new_trees_theta = np.random.uniform(0, 2 * np.pi, (N, 1))
            # on convertit en coordonnées cartésiennes et on ajoute la position du parent
            new_trees = (
                np.hstack(
                    (
                        new_trees_r * np.cos(new_trees_theta),
                        new_trees_r * np.sin(new_trees_theta),
                    )
                )
                + tree[:2]
            )
            # on concatène l'âge
            new_trees = np.hstack((new_trees, np.full((N, 1), nbyears)))
            X = np.vstack((X, new_trees))
            h = np.append(h, np.zeros(N))

        ##Delete trees that grow beyond the border
        deletable_x = np.logical_and(X[:, 0] >= 0, X[:, 0] <= 100)
        deletable_y = np.logical_and(X[:, 1] >= 0, X[:, 1] <= 100)
        delete = np.logical_and(deletable_x, deletable_y)
        h = h[delete]
        X = X[delete]

        stockage_valeurs_X.append(X[:, :2])
        stockage_valeurs_h.append(h)

    return stockage_valeurs_X, stockage_valeurs_h


from matplotlib.animation import FuncAnimation

# Création du jeu de données
nbyears = 12  # on peut modifier cette valeur, ici on choisit 10 à cause du temps que prend le programme à tourner au dessus

X, h = simForest_year_by_year(nbyears, lam, param, pmax)


def liste_matrices(X, h):
    liste_de_matrices = []
    for i in range(len(X)):
        coord = X[i]
        hauteur = h[i]
        matrice = np.full((100, 100), 0, dtype=int)
        for j in range(len(coord)):
            x, y = int(coord[j][0]), int(coord[j][1])
            matrice[x, y] = hauteur[j]
        liste_de_matrices.append(matrice)
    return liste_de_matrices


# Création de la figure
fig, ax = plt.subplots()
data = liste_matrices(X, h)
data = np.array(data)
im = ax.imshow(data[0], interpolation="nearest", cmap="Greens")


# Fonction nécessaire à l'animation
def animate(frames):
    im = ax.imshow(frames, interpolation="nearest", cmap="Greens")
    return [im]


def init():
    im.set_array(data[0])
    return [im]


def generate_frames(data, nbyears):
    index = np.array([[0,0], [0, 0]])
    frame = data[0]
    while index[1][1] <= nbyears:
        yield frame, data
        index = np.where(data == frame)
        frame = data[index[1][1] + 1]
        print(frame)


# Création de l'animation
ani = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=generate_frames(data, nbyears),
    interval=1000,
    blit=True,
    cache_frame_data=False
)
plt.show()