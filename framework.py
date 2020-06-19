import numpy as np


def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))


def derivation_of_sigmoid(z):
    return sigmoid(z) * sigmoid(1 - z)


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def derivation_of_tanh(z):
    return 1 - tanh(z) ** 2


def ReLU(z):
    return np.maximum(0, z)


def derivation_of_ReLU(z):
    return 0 if z < 0 else 1


def leaky_ReLU(z):
    return np.maximum(np.multiply(0.01, z), z)


def derivation_of_leaky_ReLU(z):
    return 0.01 if z < 0 else 1


def softmax(z):
    t = np.exp(z)
    return t / np.sum(t)


def cost(A, Y):
    return (-1 / len(Y)) * np.sum(np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), (1 - Y)))


def normalizing(X):
    mua = (1 / X.shape[1]) * np.sum(X)
    X = X - mua
    sigma = (1 / X.shape[1]) * np.sum(X ** 2)
    X /= sigma
    return X, mua, sigma


def initialization(parameters, activation_function):
    L = parameters['layer_numbers']
    for l in range(L):
        if activation_function == 'ReLU':
            parameters['W' + str(l + 1)] = np.random.randn() * np.squrt(2 / (parameters['n' + str(l)]))
        else:
            parameters['W' + str(l + 1)] = np.random.randn() * np.squrt(1 / (parameters['n' + str(l)]))
    return parameters


def gradient_descent(m, dZ, A):
    dw = (1 / m) * dZ * A.T
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    return dw, db


def gradient_descent_momentum(vdw, vdb, beta, dw, db):
    vdw = beta * vdw + dw
    vdb = beta * vdb + db
    return vdw, vdb


def RMSprop(sdw, sdb, beta, dw, db):
    sdw = beta*sdw + (1-beta)*(dw**2)
    sdb = beta * sdb + (1 - beta) * (db ** 2)
    return sdw, sdb


def Adam(sdw, sdb, vdw, vdb, beta1, beta2, dw, db):
    vdw, vdb = gradient_descent_momentum(vdw, vdb, beta1, dw, db)
    sdw, sdb = RMSprop(sdw, sdb, beta2, dw, db)
    vdwc = vdw/(1-beta1)
    vdbc = vdb/(1-beta1)
    sdwc = sdw/(1-beta2)
    sdbc = sdb/(1-beta2)
    return vdw, vdb, sdw, sdb, vdwc, vdbc, sdwc, sdbc


def forward(parameters, activation_function, last_layer_activation):
    cache = {}
    L = parameters['layer_numbers']
    for l in range(L - 1):
        cache['Z' + str(l + 1)] = parameters['W' + str(l + 1)] * cache["A" + str(l)] + parameters['b' + str(l + 1)]
        if activation_function == 'sigmoid':
            cache["A" + str(l + 1)] = sigmoid(cache['Z' + str(l + 1)])
        elif activation_function == 'tanh':
            cache["A" + str(l + 1)] = tanh(cache['Z' + str(l + 1)])
        elif activation_function == 'ReLU':
            cache["A" + str(l + 1)] = ReLU(cache['Z' + str(l + 1)])
        elif activation_function == 'leaky_ReLU':
            cache["A" + str(l + 1)] = leaky_ReLU(cache['Z' + str(l + 1)])
    cache['Z' + str(L)] = parameters['W' + str(L)] * cache["A" + str(L - 1)] + parameters['b' + str(L)]
    if last_layer_activation == 'sigmoid':
        cache['AL'] = sigmoid(cache['Z' + str(L)])
    elif last_layer_activation == 'softmax':
        cache['AL'] = softmax(cache['Z' + str(L)])
    return cache


def backward(cache, activation_function, parameters, optimization):
    m = parameters['m']
    L = parameters['layer_numbers']
    cache['dZL'] = cache['AL'] - parameters['Y']
    cache['dWL'] = (1 / m) * cache['dZL'] * cache['A' + str(L - 1)].T
    cache['dbL'] = (1 / m) * np.sum(cache['dZL'], axis=1, keepdims=True)
    Vdw, Vdb, Sdw, Sdb = 0, 0, 0, 0
    for l in reversed(range(1, L)):
        if activation_function == 'sigmoid':
            cache['dZ' + str(l)] = cache['dW' + str(l + 1)].T * cache['dZ' + str(l + 1)] * derivation_of_sigmoid(
                cache['Z' + str(l)])
        elif activation_function == 'tanh':
            cache['dZ' + str(l)] = cache['dW' + str(l + 1)].T * cache['dZ' + str(l + 1)] * derivation_of_tanh(
                cache['Z' + str(l)])
        elif activation_function == 'ReLU':
            cache['dZ' + str(l)] = cache['dW' + str(l + 1)].T * cache['dZ' + str(l + 1)] * derivation_of_ReLU(
                cache['Z' + str(l)])
        elif activation_function == 'leaky_ReLU':
            cache['dZ' + str(l)] = cache['dW' + str(l + 1)].T * cache['dZ' + str(l + 1)] * derivation_of_leaky_ReLU(
                cache['Z' + str(l)])
        dW, db = gradient_descent(m, cache['dZ' + str(l)], cache['A' + str(l - 1)])
        cache['dW' + str(l)], cache['db' + str(l)] = dW, db

        if optimization == 'gradient_descent_momentum':
            Vdw, Vdb = gradient_descent_momentum(
              Vdw, Vdb, parameters['beta'], dW, db
            )
            cache['dW' + str(l)], cache['db' + str(l)] = Vdw, Vdb

        elif optimization == 'RMSprop':
            Sdw, Sdb = RMSprop(
                Sdw, Sdb, parameters['beta'], dW, db
            )
            cache['dW' + str(l)], cache['db' + str(l)] = dW / (np.sqrt(Sdw + parameters['epsilon'])),\
                                                         Sdb / (np.sqrt(Sdb + parameters['epsilon']))

        elif optimization == 'Adam':
            Vdw, Vdb, Sdw, Sdb, vdwc, vdbc, sdwc, sdbc = Adam(
                Sdw, Sdb, Vdw, Vdb, parameters['beta1'], parameters['beta2'], dW, db
            )
            cache['dW' + str(l)], cache['db' + str(l)] = vdwc/np.sqrt(sdwc + parameters['epsilon']), \
                                                         vdbc/np.sqrt(sdbc + parameters['epsilon'])
    return cache

