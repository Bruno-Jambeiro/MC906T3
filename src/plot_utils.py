import numpy as np
import matplotlib.pyplot as plt
from estruturas import  Model
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PLOTS_DIR = "plots"

def plot_decision_boundary(model: Model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, data_name: str = ""):
    name = model.name
    X_all = X_train if X_test is None else np.vstack((X_train, X_test))
    y_all = y_train if y_test is None else np.concatenate((y_train, y_test))
    h = 0.01
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    logits = model.forward(grid_points)
    
    if logits.ndim == 2 and logits.shape[1] == 2:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
    else:
        probs = 1 / (1 + np.exp(-logits.flatten()))
        
    Z = probs.reshape(xx.shape)
    cmap_custom = LinearSegmentedColormap.from_list("BlueOrange", ["#4B8BBE", "#FCEFDD", "#F29D4B"])
    
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, levels=100, cmap=cmap_custom, alpha=0.9, vmin=0, vmax=1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_custom, edgecolors='white', linewidth=1, s=40, label='Train')
    if X_test is not None and y_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_custom, marker='X', edgecolors='black', linewidth=1.2, s=60, label='Test')
    plt.title(f"Fronteira de Decisão Modelo {name}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    
    SAVE_DIR = os.path.join(PLOTS_DIR, data_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR,f"Fronteira_{name}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_loss_curve(model, train_losses, test_losses=None, data_name: str = ""):
    plt.plot(train_losses, label='Train')
    if test_losses is not None:
        plt.plot(test_losses, label='Test')
    plt.ylim(bottom=0)
    name = model.name
    # Habilita os marcadores menores (subdivisões)
    plt.minorticks_on()
    
    # Grid principal (linhas sólidas, mais escuras)
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Grid secundário (linhas pontilhadas, mais claras)
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.title(f"Perda ao longo do treinamento - Modelo {name}")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()
    
    SAVE_DIR = os.path.join(PLOTS_DIR, data_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR,f"Losses_{name}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()