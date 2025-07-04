import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    return 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    return y + 0.1 * varNoise * np.random.normal(0, 1, len(y))

def polynomial_regression(degree, x, y_measured, train_size=0.7):
    poly = PolynomialFeatures(degree=degree)
    xnew = poly.fit_transform(x)

    y_measured = y_measured.flatten()

    np.random.seed(12)
    indeksi = np.random.permutation(len(xnew))
    train_count = int(np.floor(train_size * len(xnew)))

    indeksi_train = indeksi[:train_count]
    indeksi_test = indeksi[train_count:]

    xtrain, ytrain = xnew[indeksi_train], y_measured[indeksi_train]
    xtest, ytest = xnew[indeksi_test], y_measured[indeksi_test]

    model = lm.LinearRegression()
    model.fit(xtrain, ytrain)

    ytrain_p = model.predict(xtrain)
    ytest_p = model.predict(xtest)

    MSE_train = mean_squared_error(ytrain, ytrain_p)
    MSE_test = mean_squared_error(ytest, ytest_p)

    return model, poly, MSE_train, MSE_test

# Generiranje podataka
num_samples = 50
x = np.linspace(1, 10, num_samples)[:, np.newaxis]
y_true = non_func(x)
y_measured = add_noise(y_true)[:, np.newaxis]

degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []
models = []
polys = []

plt.figure(figsize=(8, 6))
plt.plot(x, y_true, label='Originalna funkcija', color='black', linestyle='dashed')

for degree in degrees:
    model, poly, mse_train, mse_test = polynomial_regression(degree, x, y_measured)
    MSEtrain.append(mse_train)
    MSEtest.append(mse_test)
    models.append(model)
    polys.append(poly)

    x_poly = poly.transform(x)
    y_pred = model.predict(x_poly)
    plt.plot(x, y_pred, label=f'Polinom stupnja {degree}')

plt.scatter(x, y_measured, color='gray', alpha=0.5, label='Podaci s mjerenja')
plt.xlabel('x vrijednost')
plt.ylabel('y vrijednost')
plt.title('Usporedba modela polinomske regresije')
plt.legend()
plt.grid(True)
plt.show()

print("Pogreška na testnom skupu (MSE):", MSEtest)
print("Pogreška na trening skupu (MSE):", MSEtrain)
