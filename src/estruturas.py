# O objetivo desse arquivo é criar as estruturas básicas para as redes neurais do projetos, 
# como camadas, funções de ativação, otimizadores, etc.
from abc import ABC, abstractmethod
from mimetypes import init # Para criar classes abstratas. Não é importante para a lógica do código.
import numpy as np
SEED = 260382
np.random.seed(SEED) #Garantir experimentos reprodutíveis, usando a mesma seed para todas as inicializações aleatórias.
class Layer(ABC):
    @abstractmethod
    def forward(self, X):
        pass
    @abstractmethod
    def backward(self, grad):
        pass
    @abstractmethod
    def update_weights(self, learning_rate, Ridge=0, Lasso=0):
        pass
    @abstractmethod
    def clear_weights(self):
        pass

class Optimizer(ABC):
    def __init__(self, learning_rate, scheduler=None):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.scheduler = scheduler

    @abstractmethod
    def update_weights(self, layers, Ridge=0, Lasso=0):
        pass

    def step_scheduler(self, epoch):
        if self.scheduler:
            self.learning_rate = self.scheduler.step(self.initial_learning_rate, epoch)

class LRScheduler(ABC):
    @abstractmethod
    def step(self, initial_lr, epoch):
        pass

class StepLR(LRScheduler):
    def __init__(self, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma

    def step(self, initial_lr, epoch):
        exponent = epoch // self.step_size
        return initial_lr * (self.gamma ** exponent)

class ExponentialLR(LRScheduler):
    def __init__(self, gamma=0.95):
        self.gamma = gamma

    def step(self, initial_lr, epoch):
        return initial_lr * (self.gamma ** epoch)

class LayerDense(Layer):
    def __init__(self, input_size, output_size, init = "Simple"):
        self.init = init
        self.input_size = input_size
        self.output_size = output_size
        self.init_weights(init)
        self.biases = np.zeros((1, output_size))
    def init_weights(self, init):
        if init == "Simple":
            self.weights = np.random.randn(self.input_size, self.output_size) * 2
        elif init == "Xavier":
            # Inicialização de Xavier (Glorot) - Ideal para ativações como Tanh ou Sigmoid
            # Variância = 1 / input_size
            desvio_padrao = np.sqrt(1.0 / self.input_size)
            self.weights = np.random.randn(self.input_size, self.output_size) * desvio_padrao
            
        elif init == "He":
            # Inicialização de He (Kaiming) - Ideal para ativações ReLU (O CASO DO SEU PROJETO!)
            # Variância = 2 / input_size
            desvio_padrao = np.sqrt(2.0 / self.input_size)
            self.weights = np.random.randn(self.input_size, self.output_size) * desvio_padrao
        else :
            raise NotImplementedError("Modo de inicialização não implementado.")
        
    def forward(self, X):
        self.input = X
        return np.dot(X, self.weights) + self.biases
    def backward(self, grad):
        self.grad_weights = np.dot(self.input.T, grad)
        self.grad_biases = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)
    def update_weights(self, learning_rate, Ridge=0, Lasso=0):
        self.weights -= learning_rate * self.grad_weights +  2 * Ridge * self.weights + Lasso * np.sign(self.weights)
        self.biases -= learning_rate * self.grad_biases
    def clear_weights(self):
        self.biases = np.zeros_like(self.biases)
        self.init_weights(self.init)

class Relu(Layer):
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    def backward(self, grad):
        relu_grad = self.input > 0
        return grad * relu_grad
    def update_weights(self, learning_rate, Ridge=0, Lasso=0):
        pass # ReLU não tem pesos para atualizar
    def clear_weights(self):
        pass # ReLU não tem pesos para limpar
    
class SoftmaxCrossEntropy(Layer):
    #Combina a softmax e a cross-entropy em uma única camada para melhorar a estabilidade numérica.
    def __init__(self, Ridge = 0, Lasso = 0):
        super().__init__()
        self.Ridge = Ridge
        self.Lasso = Lasso
        
    def __forward(self, X, y, model=None):
        self.input = X
        self.y_true = y
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.mean(np.log(self.probs[range(len(y)), y]))
        return loss
    def forward(self, X, y, model=None):
        loss = self.__forward(X, y, model)
        total_weights_abs = 0
        total_weights_sq = 0
        # Adicionar termos de regularização à perda
        if self.Ridge != 0:
            total_weights_sq = np.sum([np.sum(layer.weights ** 2) for layer in model.layers if hasattr(layer, 'weights')])  
            loss += self.Ridge * total_weights_sq
        if self.Lasso != 0:
            total_weights_abs = np.sum([np.sum(np.abs(layer.weights)) for layer in model.layers if hasattr(layer, 'weights')])  
            loss += self.Lasso * total_weights_abs
        
        return loss
    def backward(self, X, y, model=None):  
        loss = self.__forward(X, y, model)  # Recalcula a perda para garantir que as probabilidades estejam atualizadas
        grad = self.probs.copy()
        grad[range(len(y)), y] -= 1
        grad /= len(y)
        return grad
    def update_weights(self, learning_rate, Ridge=0, Lasso=0):
        pass # Camada de perda não tem pesos para atualizar  
    def clear_weights(self):        
        pass # Camada de perda não tem pesos para limpar
    
    
class FeatureExpansion(Layer):
    #Essa camada é responsável por expandir as características de entrada, criando novas características a partir das originais.
    #Como nosso trabalho é sobre fronteiras não lineares, essa camada pode ser bem útil para que o modelo consiga aprender essas fronteiras.
    def forward(self, X):
        self.input = X
        return np.hstack((X, X**2))
    def backward(self, grad):
        grad_input = grad[:, :self.input.shape[1]] + 2 * self.input * grad[:, self.input.shape[1]:2*self.input.shape[1]]
        return grad_input
    def update_weights(self, learning_rate, Ridge=0, Lasso=0):
        pass # Camada de expansão de características não tem pesos para atualizar
    def clear_weights(self):        
        pass # Camada de expansão de características não tem pesos para limpar
    

class SGD(Optimizer):
    def __init__(self, learning_rate, scheduler=None):
        super().__init__(learning_rate, scheduler)
    def update_weights(self, layers, Ridge=0, Lasso=0):
        for layer in layers:
            layer.update_weights(self.learning_rate, Ridge, Lasso)

class ADAM(Optimizer):
    # Extra mencionado no pdf, interessante de se fazer depois para melhorar o desempenho do modelo.
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, scheduler=None):
        super().__init__(learning_rate, scheduler)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update_weights(self, layers, Ridge=0, Lasso=0):
        self.t += 1
        beta1_t = 1 - self.beta1 ** self.t
        beta2_t = 1 - self.beta2 ** self.t

        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                if i not in self.m:
                    self.m[i] = {
                        'w': np.zeros_like(layer.weights),
                        'b': np.zeros_like(layer.biases)
                    }
                    self.v[i] = {
                        'w': np.zeros_like(layer.weights),
                        'b': np.zeros_like(layer.biases)
                    }

                grad_w = layer.grad_weights + 2 * Ridge * layer.weights + Lasso * np.sign(layer.weights)
                grad_b = layer.grad_biases

                self.m[i]['w'] = self.beta1 * self.m[i]['w'] + (1 - self.beta1) * grad_w
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * grad_b

                self.v[i]['w'] = self.beta2 * self.v[i]['w'] + (1 - self.beta2) * (grad_w ** 2)
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (grad_b ** 2)

                m_hat_w = self.m[i]['w'] / beta1_t
                m_hat_b = self.m[i]['b'] / beta1_t
                v_hat_w = self.v[i]['w'] / beta2_t
                v_hat_b = self.v[i]['b'] / beta2_t

                layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
class Model:
    def __init__(self, name: str, layers = [], loss = None, optimizer = None):
        self.name = name
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward_gradient(self, y_pred, y_true):
        grad = self.loss.backward(y_pred, y_true, self)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    def update_weights(self, ):
        self.optimizer.update_weights(self.layers, self.loss.Ridge, self.loss.Lasso)
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs = None, batch_size = None):
        train_losses = []
        train_accs = []
        test_losses = [] if X_test is not None and y_test is not None else None
        test_accs = [] if X_test is not None and y_test is not None else None

        # Função auxiliar para calcular acurácia
        def calc_accuracy(y_pred, y_true):
            predictions = np.argmax(y_pred, axis=1)
            return np.mean(predictions == y_true)

        y_pred = self.forward(X_train)
        train_losses.append(self.loss.forward(y_pred, y_train, self))
        train_accs.append(calc_accuracy(y_pred, y_train))
        if test_losses is not None:
            y_test_pred = self.forward(X_test)
            test_losses.append(self.loss.forward(y_test_pred, y_test, self))
            test_accs.append(calc_accuracy(y_test_pred, y_test))
        
        for epoch in range(1, epochs + 1):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                loss = self.loss.forward(y_pred, y_batch, self)
                grad = self.backward_gradient(y_pred, y_batch)
                self.update_weights()
            
            self.optimizer.step_scheduler(epoch)

            y_pred = self.forward(X_train)
            train_losses.append(self.loss.forward(y_pred, y_train, self))
            train_accs.append(calc_accuracy(y_pred, y_train))
            if test_losses is not None:
                y_test_pred = self.forward(X_test)
                test_losses.append(self.loss.forward(y_test_pred, y_test, self))
                test_accs.append(calc_accuracy(y_test_pred, y_test))
        return train_losses, test_losses, train_accs, test_accs
    def clear(self):
        for layer in self.layers:
            layer.clear_weights()

        self.optimizer.learning_rate = self.optimizer.initial_learning_rate

        if hasattr(self.optimizer, 'velocities'):
            self.optimizer.velocities = {}
            
        if hasattr(self.optimizer, 'm'):
            self.optimizer.m = {}
            self.optimizer.v = {}
            self.optimizer.t = 0

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9, scheduler=None):
        super().__init__(learning_rate, scheduler)
        self.beta = beta
        # Dicionário para guardar as "velocidades" de cada camada pelo índice
        self.velocities = {}
        
    def update_weights(self, layers, Ridge=0, Lasso=0):
        for i, layer in enumerate(layers):
            # Apenas camadas com pesos precisam ser atualizadas
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                # Inicializa as velocidades com zero na primeira iteração
                if i not in self.velocities:
                    self.velocities[i] = {
                        'w': np.zeros_like(layer.weights),
                        'b': np.zeros_like(layer.biases)
                    }
                
                # Resgata o gradiente atual e aplica as regularizações (L1/L2) já implementadas
                grad_w = layer.grad_weights + 2 * Ridge * layer.weights + Lasso * np.sign(layer.weights)
                grad_b = layer.grad_biases
                
                # 1. Calcula a nova velocidade (Momento)
                self.velocities[i]['w'] = self.beta * self.velocities[i]['w'] + (1 - self.beta) * grad_w
                self.velocities[i]['b'] = self.beta * self.velocities[i]['b'] + (1 - self.beta) * grad_b
                
                # 2. Atualiza os pesos reais da camada
                layer.weights -= self.learning_rate * self.velocities[i]['w']
                layer.biases -= self.learning_rate * self.velocities[i]['b']

