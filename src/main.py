import numpy as np
import matplotlib.pyplot as plt

from estruturas import LayerDense, FeatureExpansion, Relu, SoftmaxCrossEntropy, Model, SGD
from plot_utils import plot_decision_boundary, plot_loss_curve, plot_internal_decision_boundries, plot_accuracy_curve
from datasets import generate_datasets



def test_model(model:Model, data_sets):
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
        plot_internal_decision_boundries(model, X_train, Y_train, X_test, Y_test, dataset_name)
    
def define_models():
    models = []

    modelo_simples = Model("Modelo_Simples", [LayerDense(2, 6), Relu() ,LayerDense(6, 2)], SoftmaxCrossEntropy(), SGD(learning_rate=0.1))
    #ATENÇÃO, o retorno do modelo está em logits, então a função de perda já inclui a softmax.
    #O resultados podem não estar entre 0 e 1.
    models.append(modelo_simples)
    modelo_expansivo = Model("Modelo_Expansivo",[FeatureExpansion(), LayerDense(4, 6), Relu(), LayerDense(6, 2)], SoftmaxCrossEntropy(Ridge=2e-6), SGD(learning_rate=0.1))

    models.append(modelo_expansivo)
    return models

def main():
   
    models = define_models()
    full_data_sets = generate_datasets()
    for i in range(len(models)):
        test_model(models[i], full_data_sets)
if __name__ == "__main__":
    main()