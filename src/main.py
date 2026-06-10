import numpy as np
import matplotlib.pyplot as plt

from estruturas import LayerDense, FeatureExpansion, Relu, SoftmaxCrossEntropy, Model, SGD, SGDMomentum, ADAM, StepLR, ExponentialLR
from plot_utils import plot_decision_boundary, plot_loss_curve, plot_internal_decision_boundries, plot_accuracy_curve, plot_neuron_ablation_tv
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
            epochs=3000,
            batch_size=32,
        )
        plot_loss_curve(model, train_losses, test_losses, dataset_name)
        plot_accuracy_curve(model, train_accs, test_accs, dataset_name)
        plot_decision_boundary(model, X_train, Y_train, X_test, Y_test, dataset_name)
        if (PLOT_INTERNAL_BOUNDARIES):
            plot_internal_decision_boundries(model, X_train, Y_train, X_test, Y_test, dataset_name)
            
            for layer_index, layer in enumerate(model.layers):
                # A ablação só faz sentido em camadas que possuem pesos e vieses (LayerDense)
                if hasattr(layer, 'weights'):
                    
                    # Descomente o bloco abaixo se NÃO quiser ablatar a última camada (os 2 neurônios de saída das classes).
                    # Ablatar a última camada geralmente não traz informações sobre representação interna.
                    # if layer_index == len(model.layers) - 1:
                    #     continue
                    
                    num_neurons = layer.output_size
                    for neuron_index in range(num_neurons):
                        print(f"  -> Processando e salvando: Camada {layer_index}, Neurônio {neuron_index}/{num_neurons-1}")
                        
                        # Certifique-se de que a função abaixo esteja importada no início do main.py
                        # (ex: from plot_utils import plot_neuron_ablation_tv)
                        plot_neuron_ablation_tv(
                            model=model, 
                            X=X_train, # Usamos o X_train para definir os limites do grid 2D
                            y=Y_train,
                            layer_index=layer_index, 
                            neuron_index=neuron_index, 
                            dataset_name=dataset_name
                        )
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
    # #ATENÇÃO, o retorno do modelo está em logits, então a função de perda já inclui a softmax.
    # #O resultados podem não estar entre 0 e 1.
    # models.append(modelo_simples)
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
   
    models = define_models() + define_experimental_models()
    full_data_sets = generate_datasets()
    for model in models:
        print(f"Treinando e avaliando o modelo: {model.name}")
        test_model(model, full_data_sets)
if __name__ == "__main__":
    main()