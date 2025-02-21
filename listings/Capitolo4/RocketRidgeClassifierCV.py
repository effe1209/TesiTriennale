import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from scipy.special import softmax

# Genera kernel convoluzionali casuali
input_length = X_train.shape[1]
num_kernels = 10000
rocket_transformer = Rocket(num_kernels = num_kernels, n_jobs=-1)

# Applica i kernel alle serie temporali
features_train = rocket_transformer.fit_transform(X_train)
features_test = rocket_transformer.transform(X_test)

# Addestramento del modello supervisionato
model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
model.fit(features_train, y_train)

# Predizione delle anomalie nei dati di test
y_pred = model.predict(features_test)

# Per separare multiclasse o monoclasse
if  len(np.unique(y_test)) > 2:
    y_proba = softmax(model.decision_function(features_test), axis=1)
else:
    y_proba = softmax(model.decision_function(features_test), axis=0)