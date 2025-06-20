import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

def non_func(x):
    y = 1.6345 - 0.6235 * np.cos(0.6067 * x) - 1.3501 * np.sin(0.6067 * x) \
        - 1.1622 * np.cos(2 * 0.6067 * x) - 0.9443 * np.sin(2 * 0.6067 * x)
    return y

def add_noise(y):
    np.random.seed(14)
    var_noise = np.ptp(y)  # ptp = max-min
    y_noisy = y + 0.1 * var_noise * np.random.randn(len(y))
    return y_noisy

x = np.linspace(1, 10, 100)
y_true = non_func(x)
y_measured = add_noise(y_true)

plt.figure()
plt.plot(x, y_measured, 'ok', label='Mjereni podaci')
plt.plot(x, y_true, label='Originalna funkcija')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()

np.random.seed(12)
indices = np.random.permutation(len(x))
train_size = int(0.7 * len(x))
indices_train = indices[:train_size]
indices_test = indices[train_size:]

x = x.reshape(-1, 1)
y_measured = y_measured.reshape(-1, 1)

x_train = x[indices_train]
y_train = y_measured[indices_train]
x_test = x[indices_test]
y_test = y_measured[indices_test]

plt.figure()
plt.plot(x_train, y_train, 'bo', label='Trening')
plt.plot(x_test, y_test, 'ro', label='Test')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()

model = lm.LinearRegression()
model.fit(x_train, y_train)

print('Model oblika: y_hat = Theta0 + Theta1 * x')
print(f'y_hat = {model.intercept_[0]:.4f} + {model.coef_[0][0]:.4f} * x')

y_pred = model.predict(x_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f'Srednja kvadratna pogreška (MSE) na test skupu: {mse_test:.5f}')

plt.figure()
plt.plot(x_test, y_pred, 'go', label='Predikcija')
plt.plot(x_test, y_test, 'ro', label='Test podaci')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()
