import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = (1.6345 - 0.6235 * np.cos(0.6067 * x) - 1.3501 * np.sin(0.6067 * x)
         - 1.1622 * np.cos(2 * 0.6067 * x) - 0.9443 * np.sin(2 * 0.6067 * x))
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1 * varNoise * np.random.normal(0, 1, len(y))
    return y_noisy

x = np.linspace(1, 10, 50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

poly = PolynomialFeatures(degree=15)
xnew = poly.fit_transform(x)

np.random.seed(12)
indeksi = np.random.permutation(len(xnew))
cutoff = int(np.floor(0.7 * len(xnew)))

indeksi_train = indeksi[:cutoff]
indeksi_test = indeksi[cutoff + 1:]

x_train = xnew[indeksi_train]
y_train = y_measured[indeksi_train]
x_test = xnew[indeksi_test]
y_test = y_measured[indeksi_test]

model = lm.LinearRegression()
model.fit(x_train, y_train)

y_test_pred = model.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)

plt.figure()
plt.plot(x_test[:, 1], y_test_pred, 'og', label='predicted')
plt.plot(x_test[:, 1], y_test, 'or', label='test')
plt.legend(loc=4)
plt.show()

plt.figure()
plt.plot(x, y_true, label='f')
plt.plot(x, model.predict(xnew), 'r-', label='model')
plt.plot(x_train[:, 1], y_train, 'ok', label='train')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()
