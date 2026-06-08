import numpy as np
import matplotlib.pyplot as plt

from estruturas import LayerDense, FeatureExpansion, Relu, SoftmaxCrossEntropy, Model, SGD, SGDMomentum, ADAM, StepLR, ExponentialLR
from plot_utils import plot_decision_boundary, plot_loss_curve, plot_internal_decision_boundries, plot_accuracy_curve
from datasets import generate_datasets

PLOT_INTERNAL_BOUNDARIES: bool = True

def test_model(model:Model, data_sets):
    global PLOT_INTERNAL_BOUNDRIES
    for X_train, Y_train, X_test, Y_test, dataset_name in data_sets:
        model.clear()
        train_losses, test_losses, train_accs, test_accs = model.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=1000,
            batch_size=32,
        )
        plot_loss_curve(model, train_losses, test_losses, dataset_name)
        plot_accuracy_curve(model, train_accs, test_accs, dataset_name)
        plot_decision_boundary(model, X_train, Y_train, X_test, Y_test, dataset_name)
        if (PLOT_INTERNAL_BOUNDARIES):
            plot_internal_decision_boundries(model, X_train, Y_train, X_test, Y_test, dataset_name)
        else:
            print("Plot Internal Boundary Disabled")
    
def define_models():
    models = []

    modelo_simples = Model(
        "Modelo_Simples",
        [LayerDense(2, 6, init="He"), Relu() ,LayerDense(6, 2, init="He")],
        SoftmaxCrossEntropy(),
        SGD(learning_rate=0.1)
    )
    #ATENÇÃO, o retorno do modelo está em logits, então a função de perda já inclui a softmax.
    #O resultados podem não estar entre 0 e 1.
    models.append(modelo_simples)

    modelo_expansivo = Model(
        "Modelo_Expansivo",
        [FeatureExpansion(), LayerDense(4, 6, init="He"), Relu(), LayerDense(6, 2, init="He")],
        SoftmaxCrossEntropy(Ridge=2e-6),
        SGD(learning_rate=0.1)
    )
    models.append(modelo_expansivo)

    modelo_momentum = Model(
        "Modelo_Momentum", 
        [LayerDense(2, 6), Relu(), LayerDense(6, 2)], 
        SoftmaxCrossEntropy(), 
        SGDMomentum(learning_rate=0.1, beta=0.9, scheduler=StepLR(step_size=200, gamma=1.0))
    )
    models.append(modelo_momentum)

    return models

def define_experimental_models():
    models = []

    #definindo alguns modelos para avaliar o que acontece conforme a estrutura da rede neural muda
    modelo_under = Model("Modelo_Underfitting",
    [
        LayerDense(2, 3, init="He"), 
        Relu(), 
        LayerDense(3, 2, init="He")
    ],
    SoftmaxCrossEntropy(), SGD(learning_rate=0.1))
    # models.append(modelo_under)

    modelo_ideal = Model("Modelo_Ideal",
    [
        LayerDense(2, 6, init="He"), 
        Relu(), 
        LayerDense(6, 2, init="He")
    ],
    SoftmaxCrossEntropy(), SGD(learning_rate=0.1))
    # models.append(modelo_ideal)

    modelo_over = Model("Modelo_Overfitting",
    [
        LayerDense(2, 16, init="He"), 
        Relu(), 
        LayerDense(16, 16, init="He"), 
        Relu(), 
        LayerDense(16, 2, init="He")
    ],
    SoftmaxCrossEntropy(), SGD(learning_rate=0.1))
    # models.append(modelo_over)

    modelo_wide_shallow = Model("Modelo_Wide_Shallow1x20",
    [
        LayerDense(2, 20, init="He"), 
        Relu(), 
        LayerDense(20, 20, init="He"), 
        Relu(), 
        LayerDense(20, 2, init="He")
    ],
    SoftmaxCrossEntropy(), ADAM(learning_rate=0.001, scheduler=StepLR(step_size=200, gamma=0.5)))
    models.append(modelo_wide_shallow)

    modelo_balanced = Model("Modelo_Balanced2x10",
    [
        LayerDense(2, 10, init="He"), 
        Relu(), 
        LayerDense(10, 10, init="He"), 
        Relu(),
        LayerDense(10, 10, init="He"), 
        Relu(), 
        LayerDense(10, 2, init="He")
    ],
    SoftmaxCrossEntropy(), ADAM(learning_rate=0.001, scheduler=StepLR(step_size=200, gamma=0.5)))
    models.append(modelo_balanced)

    modelo_narrow_deep = Model("Modelo_Narrow_Deep4x5",
    [
        LayerDense(2, 5, init="He"), 
        Relu(), 
        LayerDense(5, 5, init="He"), 
        Relu(),
        LayerDense(5, 5, init="He"), 
        Relu(), 
        LayerDense(5, 5, init="He"), 
        Relu(), 
        LayerDense(5, 5, init="He"), 
        Relu(), 
        LayerDense(5, 2, init="He")
    ],
    SoftmaxCrossEntropy(), ADAM(learning_rate=0.001, scheduler=StepLR(step_size=200, gamma=0.5)))
    models.append(modelo_narrow_deep)

    return models

def main():
   
    models = define_experimental_models()
    full_data_sets = generate_datasets()
    for model in models:
        test_model(model, full_data_sets)
if __name__ == "__main__":
    main()