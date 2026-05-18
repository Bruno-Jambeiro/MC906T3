import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from estruturas import LayerDense, FeatureExpansion, Relu, SoftmaxCrossEntropy, Model, SGD
from plot_utils import plot_decision_boundary, plot_loss_curve
from extra_datasets import spiral_2d

def generate_datasets():
    data_sets = []
    X_circles, Y_Circles = make_circles(n_samples= 200, noise=0.2, factor=0.3, random_state= 260382) #Seed  para garantir resultados consistentes(RA de um membro do grupo)
    X_train_circles, X_test_circles, Y_train_circles, Y_test_circles = train_test_split(
        X_circles, Y_Circles, test_size=0.2, random_state=260382
    )
    data_sets.append((X_train_circles, Y_train_circles, X_test_circles, Y_test_circles, "Circles"))
    
    X_moons, Y_Moons = make_moons(n_samples= 200, noise=0.0, random_state= 260382) #Seed para garantir resultados consistentes(RA de um membro do grupo)
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



def test_model(model:Model, data_sets):
    for X_train, Y_train, X_test, Y_test, dataset_name in data_sets:
        train_losses, test_losses = model.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=1000,
            batch_size=32,
        )
        plot_loss_curve(model, train_losses, test_losses, dataset_name)
        plot_decision_boundary(model, X_train, Y_train, X_test, Y_test, dataset_name)

def main():
    models = []

    modelo_simples = Model("Modelo_Simples", [LayerDense(2, 10), Relu(), LayerDense(10, 2)], SoftmaxCrossEntropy(), SGD(learning_rate=0.1))
    #ATENÇÃO, o retorno do modelo está em logits, então a função de perda já inclui a softmax.
    #O resultados podem não estar entre 0 e 1.
    models.append(modelo_simples)
    modelo_expansivo = Model("Modelo_Expansivo",[FeatureExpansion(), LayerDense(4, 10), Relu(), LayerDense(10, 2)], SoftmaxCrossEntropy(), SGD(learning_rate=0.1))

    models.append(modelo_expansivo)

    full_data_sets = generate_datasets()
    for i in range(len(models)):
        test_model(models[i], full_data_sets)
if __name__ == "__main__":
    main()