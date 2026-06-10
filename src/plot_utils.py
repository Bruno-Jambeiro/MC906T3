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
    x_min, x_max = X_all[:, 0].min() - 0.25, X_all[:, 0].max() + 0.25
    y_min, y_max = X_all[:, 1].min() - 0.25, X_all[:, 1].max() + 0.25
    
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
    
    SAVE_DIR = os.path.join(PLOTS_DIR, data_name, name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR,f"Fronteira_{name}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_internal_decision_boundries(model: Model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, data_name: str = ""):
    # Plot a fronteira de decisão interna do modelo, mostrando a saída de cada neurônio em cada camada.
    X = X_train if X_test is None else np.vstack((X_train, X_test))
    #y = y_train if y_test is None else np.concatenate((y_train, y_test))

    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    span = max(x_max - x_min, y_max - y_min)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_min, x_max = x_mid - span / 2, x_mid + span / 2
    y_min, y_max = y_mid - span / 2, y_mid + span / 2
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    activations = [grid]
    layer_names = ['Input']
    out = grid
    for layer in model.layers:
        out = layer.forward(out)
        activations.append(out if out.ndim == 2 else out.reshape(-1, 1))
        layer_names.append(layer.__class__.__name__)

    soft = out
    if soft.ndim == 2 and soft.shape[1] == 2:
        exp_scores = np.exp(soft - np.max(soft, axis=1, keepdims=True))
        soft = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        soft = 1 / (1 + np.exp(-soft.flatten()))
        soft = soft.reshape(-1, 1)
    activations.append(soft)
    layer_names.append('Softmax')

    n_layers = len(activations)
    max_neurons = max(act.shape[1] for act in activations)
    square = 3.0
    width = max(16, square * n_layers)
    height = max(9, square * max_neurons)
    fig, axes = plt.subplots(max_neurons, n_layers, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(top=0.88, hspace=0.15, wspace=0.16)
    cmap_custom = LinearSegmentedColormap.from_list("BlueOrange", ["#4B8BBE", "#FCEFDD", "#F29D4B"])

    for col, act in enumerate(activations):
        neurons = act.shape[1]
        start = (max_neurons - neurons) // 2
        

        for row in range(max_neurons):
            ax = axes[row][col]
            if row < start or row >= start + neurons:
                ax.axis('off')
                continue
            Z = act[:, row - start].reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=100, cmap=cmap_custom, alpha=0.9, vmin=-1.0, vmax=1.0)
            #ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_custom, edgecolors='white', s=16, linewidth=0.3, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')

        header_ax = axes[start][col]
        header_ax.set_title(f'Camada {col + 1}: {layer_names[col]}', pad=12, fontsize=10)

    for col in range(n_layers):
        for row in range(max_neurons):
            if axes[row][col].lines or axes[row][col].collections:
                break
            axes[row][col].axis('off')

    plt.suptitle(f'Fronteiras Internas por Neurônio — Modelo {model.name}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    SAVE_DIR = os.path.join(PLOTS_DIR, data_name, model.name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f'InternalBoundaries_{model.name}.png'), dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_loss_curve(model, train_losses, test_losses=None, data_name: str = ""):
    #Plota a curva de perda ao longo do treinamento, tanto para os dados de treino quanto para os de teste (se fornecidos).
    #Escala logaritmica no eixo das épcoas para melhor visualização, já que a perda pode diminuir rapidamente no início do treinamento.
    plt.plot(train_losses, label='Train')
    if test_losses is not None:
        plt.plot(test_losses, label='Test')
    plt.xscale('log')
    plt.ylim(bottom=0, top=1.0)
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
    
    SAVE_DIR = os.path.join(PLOTS_DIR, data_name, name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR,f"Losses_{name}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_accuracy_curve(model, train_accs, test_accs=None, data_name: str = ""):
    # Plota a curva de acurácia ao longo do treinamento
    plt.plot(train_accs, label='Train')
    if test_accs is not None:
        plt.plot(test_accs, label='Test')
        
    # A escala para acurácia é linear, de 0 a 1 (0% a 100%)
    plt.ylim(bottom=0.0, top=1.05)
    
    name = model.name
    # Habilita os marcadores menores (subdivisões)
    plt.minorticks_on()
    
    # Grid principal e secundário
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.title(f"Acurácia ao longo do treinamento - Modelo {name}")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    
    SAVE_DIR = os.path.join(PLOTS_DIR, data_name, name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f"Accuracy_{name}.png"), dpi=300, bbox_inches='tight')

    # Salva um resumo em texto com as últimas acurácias de treino e teste
    try:
        last_train = float(train_accs[-1]) if len(train_accs) > 0 else float('nan')
    except Exception:
        last_train = float('nan')

    if test_accs is not None and len(test_accs) > 0:
        try:
            last_test = float(test_accs[-1])
        except Exception:
            last_test = float('nan')
    else:
        last_test = None

    summary_path = os.path.join(SAVE_DIR, f'Accuracy_summary_{name}.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Train_last_accuracy: {last_train:.6f}\n')
        if last_test is None:
            f.write('Test_last_accuracy: N/A\n')
        else:
            f.write(f'Test_last_accuracy: {last_test:.6f}\n')

    plt.show()
    plt.close()


def softmax_probs(logits):
    """Estabiliza e calcula a probabilidade (mesma lógica da sua Loss)."""
    exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def plot_neuron_ablation_tv(model, X, y=None, layer_index=None, neuron_index=None, dataset_name=None):
    """
    Gera um Heatmap da TV Distance mostrando a região de responsabilidade 
    de um neurônio específico, sem modificar a estrutura do modelo.
    """
    # 1. Definir os limites do Grid com base nos dados (similar ao seu plot_decision_boundary)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 2. Obter as probabilidades do modelo INTACTO (Baseline)
    logits_orig = model.forward(grid)
    P_orig = softmax_probs(logits_orig) # Shape: (N_pontos, 2 classes)

    # 3. Acessar a camada alvo (precisa ser uma LayerDense)
    target_layer = model.layers[layer_index]
    if not hasattr(target_layer, 'weights'):
        raise ValueError("A camada especificada não possui pesos (não é LayerDense).")

    # 4. CIRURGIA DE ABLAÇÃO (Backup -> Zerar -> Prever -> Restaurar)
    # Fazer backup físico (copy) para não perder os ponteiros
    w_backup = np.copy(target_layer.weights[:, neuron_index])
    b_backup = np.copy(target_layer.biases[0, neuron_index])

    try:
        # Forçar a morte do neurônio
        target_layer.weights[:, neuron_index] = 0
        target_layer.biases[0, neuron_index] = 0

        # Obter probabilidades do modelo ABLATADO
        logits_abl = model.forward(grid)
        P_abl = softmax_probs(logits_abl)

    finally:
        # Garantir a restauração incondicional dos pesos originais
        target_layer.weights[:, neuron_index] = w_backup
        target_layer.biases[0, neuron_index] = b_backup

    # 5. Calcular a Distância de Variação Total (TV Distance)
    # TV = 1/2 * soma_sobre_classes( | P_orig - P_abl | )
    D_TV = 0.5 * np.sum(np.abs(P_orig - P_abl), axis=1)
    D_TV = D_TV.reshape(xx.shape)

    # 6. Plotar o Heatmap
    plt.figure(figsize=(8, 6))
    # Usamos o colormap 'Reds'. Branco = 0 deformação, Vermelho Escuro = alta deformação 
    contour = plt.contourf(xx, yy, D_TV, levels=20, cmap='Reds', vmin=0.0, vmax=1.0)
    plt.colorbar(contour, label='TV Distance (Degradação da Probabilidade)')
    
    # Plotar o dataset em background com transparência para dar contexto espacial
    # Se labels (`y`) forem fornecidas, colorimos por classe; caso contrário,
    # usamos as predições do modelo intacto para inferir a classe.
    if y is None:
        inferred_y = np.argmax(P_orig, axis=1)
    else:
        inferred_y = np.ravel(y)

    # Usar azul para a classe 0 e laranja para a classe 1
    class_cmap = LinearSegmentedColormap.from_list("BlueOrangeClasses", ["#4B8BBE", "#F29D4B"])
    plt.scatter(X[:, 0], X[:, 1], c=inferred_y, cmap=class_cmap, edgecolors='k', alpha=0.3, s=20)
    
    plt.title(f'Ablação: Camada {layer_index}, Neurônio {neuron_index}\n{model.name} - {dataset_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    SAVE_DIR = os.path.join(PLOTS_DIR, dataset_name, model.name, "Ablation")
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f'Ablação:Camada_{layer_index}_N{neuron_index}_{model.name}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()