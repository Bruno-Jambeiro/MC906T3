import numpy as np
from estruturas import LayerDense, Relu, SoftmaxCrossEntropy, Model, SGD
from datasets import generate_datasets

def realizar_grid_search_completo():
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    batch_sizes = [8, 16, 32, 64]
    
    datasets = generate_datasets()
    
    for X_train, y_train, X_test, y_test, dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"INICIANDO BUSCA NO DATASET: {dataset_name.upper()}")
        print(f"{'='*50}")
        
        melhor_acc = 0
        melhor_config = None
        
        for lr in learning_rates:
            for bs in batch_sizes:
                # Criação do modelo limpo para cada teste
                modelo = Model(
                    f"Simples_{dataset_name}_LR_{lr}_BS_{bs}",
                    [LayerDense(2, 6, init="He"), Relu(), LayerDense(6, 2, init="He")],
                    SoftmaxCrossEntropy(),
                    SGD(learning_rate=lr)
                )
                
                # Treinamento
                train_losses, test_losses, train_accs, test_accs = modelo.train(
                    X_train, y_train, X_test, y_test, epochs=400, batch_size=bs
                )
                
                # Média das últimas 10 épocas para estabilidade
                acc_final_teste = np.mean(test_accs[-10:])
                
                print(f"LR: {lr:<5} | Batch: {bs:<3} | Acurácia: {acc_final_teste:.4f}")
                
                if acc_final_teste > melhor_acc:
                    melhor_acc = acc_final_teste
                    melhor_config = (lr, bs)
                    
        print(f"\nMELHOR CONFIGURAÇÃO PARA {dataset_name.upper()}")
        print(f"LR: {melhor_config[0]} | Batch Size: {melhor_config[1]} | Acurácia: {melhor_acc:.4f}")

if __name__ == "__main__":
    realizar_grid_search_completo()