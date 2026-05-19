import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

def spiral_2d(n_samples, noise, rotations=2.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples_per_class = n_samples // 2
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    
    # Raiz quadrada suaviza a densidade, evitando um "borrão" concentrado na origem
    base_t = np.sqrt(np.random.rand(n_samples_per_class)) * (rotations * 2 * np.pi)
    
    for j in range(2):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        
        # O raio cresce proporcionalmente ao ângulo
        r = base_t / (rotations * 2 * np.pi)
        
        # O pulo do gato: A Classe 0 tem defasagem 0. A Classe 1 tem defasagem de PI.
        t = base_t + (j * np.pi)
        
        # Conversão polar para cartesiana + Ruído espacial isotrópico (Gaussian)
        X[ix, 0] = r * np.cos(t) + np.random.randn(n_samples_per_class) * noise
        X[ix, 1] = r * np.sin(t) + np.random.randn(n_samples_per_class) * noise
        y[ix] = j
        
    return X, y


def generate_datasets():
    data_sets = []
    X_circles, Y_Circles = make_circles(n_samples= 200, noise=0.2, factor=0.3, random_state= 260382) #Seed  para garantir resultados consistentes(RA de um membro do grupo)
    X_train_circles, X_test_circles, Y_train_circles, Y_test_circles = train_test_split(
        X_circles, Y_Circles, test_size=0.2, random_state=260382
    )
    data_sets.append((X_train_circles, Y_train_circles, X_test_circles, Y_test_circles, "Circles"))
    
    X_moons, Y_Moons = make_moons(n_samples= 200, noise=0.1, random_state= 260382) #Seed para garantir resultados consistentes(RA de um membro do grupo)
    X_train_moons, X_test_moons, Y_train_moons, Y_test_moons = train_test_split(
        X_moons, Y_Moons, test_size=0.2, random_state=260382
    )
    data_sets.append((X_train_moons, Y_train_moons, X_test_moons, Y_test_moons, "Moons"))
    
    X_spiral, Y_Spiral = spiral_2d(n_samples=500, noise=0.01, random_state=260382) #Seed para garantir resultados consistentes(RA de um membro do grupo)
    X_train_spiral, X_test_spiral, Y_train_spiral, Y_test_spiral = train_test_split(
        X_spiral, Y_Spiral, test_size=0.2, random_state=260382
    )
    data_sets.append((X_train_spiral, Y_train_spiral, X_test_spiral, Y_test_spiral, "Spiral"))
    return data_sets