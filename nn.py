# As background visualizer we train a small NN on synthetic 2D classification data
# 
import numpy as np
import itertools
from pyodide.ffi import to_js
# from js import drawPoints, document
# n = 4
def generate():
    global x, y
    n = 1000
    x = np.random.rand(n, 2)
    y = np.zeros(n)
    # The center of a next blob should be within 2*radius of the previous one
    n_blobs = 10
    for i in range(n_blobs):
        if i == 0:
            # For first blob we want it near the center
            center = np.random.rand(2) * 0.5 + 0.25
        else:
            center = center + np.random.rand(2) * 3 * radius - radius
        radius = np.random.rand() * 0.3
        y += np.exp(-np.sum((x - center)**2, axis=1) / radius**2)
    # Normalize y
    y = (y > 0.5).astype(int)
    # Put into one array with 3 columns
    data = np.hstack([x, y.reshape(-1, 1)])
    return to_js(data.tolist())

class Sigmoid():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        y = 1 / (1 + np.exp(-x)) # (batch_size, in_size)
        self.y = y
        return y
    
    def backward(self, dLdy, opt):
        return dLdy * self.y * (1 - self.y) # (batch_size, in_size) * (batch_size, in_size) = (batch_size, in_size)
    
class ReLU():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dLdy, opt):
        mask = self.x > 0 # (batch_size, in_size)
        return dLdy * mask # (batch_size, in_size) * (batch_size, in_size) = (batch_size, in_size)
    
class GELU():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # Approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) # (batch_size, in_size)
    
    def backward(self, dLdy, opt):
        x = self.x
        # Approximation
        s = x / np.sqrt(2)
        erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2))
        approx = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
        dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / np.sqrt(2))
        return dLdy * dx # (batch_size, in_size) * (batch_size, in_size) = (batch_size, in_size)
    
class Linear():
    id_iter = itertools.count()
    def __init__(self, in_size, out_size):
        self.x = None
        self.id_w = next(Linear.id_iter)
        self.id_b = next(Linear.id_iter)
        self.in_size = in_size
        self.out_size = out_size
        # self.weights = np.random.randn(in_size, out_size) # (in_size, out_size)
        # self.weights /= in_size # Normalize
        # self.bias = np.zeros(out_size) # (out_size)
        # Better initialization
        stdv = 1. / np.sqrt(in_size)
        self.weights = np.random.uniform(-stdv, stdv, (in_size, out_size))
        self.bias = np.random.uniform(-stdv, stdv, out_size)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias # (batch_size, in_size) @ (batch_size, in_size, out_size) = (batch_siz,e out_size)
    
    def backward(self, dLdy, opt):
        dLdw = self.x.T @ dLdy # (batch_size, in_size)^T @ (batch_size, out_size) = (in_size, out_size)
        dLdb = dLdy.sum() # (batch_size, out_size) -> (out_size)
        self.weights = opt.update(self.id_w, self.weights, dLdw)
        self.bias = opt.update(self.id_b, self.bias, dLdb)
        dLdx = dLdy @ self.weights.T # (batch_size, out_size) @ (in_size, out_size)^T = (batch_size, in_size)
        return dLdx
    
class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dLdy, opt):
        for layer in self.layers[::-1]:
            dLdy = layer.backward(dLdy, opt)
        return dLdy
    
class BinaryCrossEntropy():
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t): # y: predicted, t: target
        t = t.reshape(-1, 1) if len(t.shape) == 1 else t # (batch_size) -> (batch_size, 1) if needed
        self.y = y
        self.t = t
        return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y)) # (batch_size, out_size) -> (1)
    
    def backward(self):
        return (self.y - self.t) / (self.y * (1 - self.y)) / self.t.size # (batch_size, out_size) - (batch_size, out_size) = (batch_size, out_size) (Normalized by batch size)
    
class AdamW():
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.001):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, id, param, grad):
        self.t[id] = self.t.get(id, 0) + 1 if self.t.get(id, 0) < 5000 else 1
        t = self.t[id]
        if id not in self.m:
            self.m[id] = np.zeros_like(param)
            self.v[id] = np.zeros_like(param)
        param = param - self.lr * self.weight_decay * param
        m = self.m[id]
        v = self.v[id]
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.m[id] = m
        self.v[id] = v
        return param

def initialize_model():
    print("Initializing model")
    global classifier, criterion, opt
    hidden_size = 8
    # Randomize between relu and gelu
    act = 'ReLU' if np.random.rand() < 0.5 else 'GELU'
    classifier = NeuralNetwork([
        Linear(2, hidden_size),
        globals()[act](),
        Linear(hidden_size, hidden_size),
        globals()[act](),
        Linear(hidden_size, 1),
        Sigmoid()
    ])
    criterion = BinaryCrossEntropy()
    opt = AdamW(0.01, weight_decay=0.0001)

i = 0

def step():
    global classifier, criterion, opt, x, y, i
    y_pred = classifier.forward(x)
    l = criterion.forward(y_pred, y)
    dLdy = criterion.backward()
    classifier.backward(dLdy, opt)
    # print(f"Loss: {l}")
    # Get the decision boundary
    x1 = np.linspace(0, 1, 72)
    x2 = np.linspace(0, 1, 72)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y_p = classifier.forward(X).reshape(-1)
    # boundary line is just where y_pred is between 0.4 and 0.6
    b_line = X[(y_p > 0.2) & (y_p < 0.8)]
    y_pred = y_pred.reshape(-1) > 0.5
    accuracy = np.mean(y_pred == y)
    i += 1
    if i % 50 == 0:
        print(f"Iteration: {i}, Loss: {l}, Accuracy: {accuracy}")
    # if loss is nan, reset the model
    if np.isnan(l):
        initialize_model()
    return to_js(b_line.tolist())