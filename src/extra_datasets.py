import numpy as np


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