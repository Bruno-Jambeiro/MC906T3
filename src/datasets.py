import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
import platform
from PIL import Image, ImageDraw, ImageFont

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


def text_2d(text_line1="MC", text_line2="906", n_samples=500, noise=0.01, random_state=None, width=800, height=800):
    """
    Gera um dataset com texto em preto sobre fundo branco de forma vetorizada.
    
    Classe 0: Fundo branco
    Classe 1: Letras em preto
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Gerenciamento robusto de fontes cross-platform
    try:
        os_name = platform.system()
        if os_name == "Windows":
            font = ImageFont.truetype("arial.ttf", 400)
        elif os_name == "Darwin": # macOS
            font = ImageFont.truetype("Arial.ttf", 400)
        else: # Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 400)
    except OSError:
        print("AVISO CRÍTICO: Fonte TrueType não encontrada. O texto usará o fallback e ficará ilegível.")
        font = ImageFont.load_default()
    
    # Desenhar primeira linha
    bbox1 = draw.textbbox((0, 0), text_line1, font=font)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    y1 = height // 4 - (bbox1[3] - bbox1[1]) // 2
    y1 = max(0, min(y1, height - (bbox1[3] - bbox1[1])))
    draw.text((x1, y1), text_line1, fill=0, font=font)
    
    # Desenhar segunda linha
    bbox2 = draw.textbbox((0, 0), text_line2, font=font)
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    y2 = height * 3 // 4 - (bbox2[3] - bbox2[1]) // 2 - 50
    y2 = max(0, min(y2, height - (bbox2[3] - bbox2[1]) - 50))
    draw.text((x2, y2), text_line2, fill=0, font=font)
    
    img_array = np.array(img)
    
    # VETORIZAÇÃO: Criar matrizes de coordenadas X e Y
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    
    # Normalização isotrópica (usando o mesmo fator de escala para manter o aspect ratio 4:1)
    scale = width / 2
    x_norm = (j - width / 2) / scale
    
    # Correção crucial: Inverter o sinal do eixo Y para compatibilidade cartesiana
    y_norm = -(i - height / 2) / scale 
    
    # Separação por máscaras booleanas (Thresholding vetorizado)
    mask_0 = img_array > 127
    mask_1 = img_array <= 127
    
    class_0_points = np.column_stack((x_norm[mask_0], y_norm[mask_0]))
    class_1_points = np.column_stack((x_norm[mask_1], y_norm[mask_1]))
    
    # Determinar número de amostras
    if n_samples is not None:
        samples_per_class = n_samples // 2
    else:
        samples_per_class = min(len(class_0_points), len(class_1_points))
        
    samples_per_class = min(samples_per_class, len(class_0_points), len(class_1_points))
    
    # Amostragem (sem reposição)
    indices_0 = np.random.choice(len(class_0_points), samples_per_class, replace=False)
    indices_1 = np.random.choice(len(class_1_points), samples_per_class, replace=False)
    
    X_0 = class_0_points[indices_0]
    X_1 = class_1_points[indices_1]
    
    # Adicionar ruído
    if noise > 0:
        X_0 += np.random.randn(*X_0.shape) * noise
        X_1 += np.random.randn(*X_1.shape) * noise
        
    # Combinação e embaralhamento final
    X = np.vstack([X_0, X_1])
    y_labels = np.hstack([np.zeros(samples_per_class, dtype=int), 
                          np.ones(samples_per_class, dtype=int)])
    
    permutation = np.random.permutation(len(X))
    return X[permutation], y_labels[permutation]


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
    
    X_text, Y_Text = text_2d( n_samples=2000, noise=0.00, random_state=260382) #Seed para garantir resultados consistentes(RA de um membro do grupo)
    X_train_text, X_test_text, Y_train_text, Y_test_text = train_test_split(
        X_text, Y_Text, test_size=0.05, random_state=260382
    )
    data_sets.append((X_train_text, Y_train_text, X_test_text, Y_test_text, "MC906"))
    return data_sets