from sktime.transformations.panel.rocket import Rocket
import numpy as np

def detect_anomalies_with_threshold(scores, threshold):
    return (scores > threshold).astype(int)

# Genera kernel convoluzionali casuali
input_length = X_train.shape[1]
num_kernels = 10000

rocket_transformer = Rocket(num_kernels = num_kernels, n_jobs=-1)

# Applica i kernel alle serie temporali
features_train = rocket_transformer.fit_transform(X_train)
features_test = rocket_transformer.transform(X_test)

# Sintesi delle caratteristiche per esempio
anomaly_scores_train = np.mean(features_train, axis=1)  
anomaly_scores_test = np.mean(features_test, axis=1)  

# Rilevamento delle anomalie
threshold = np.percentile(anomaly_scores_train , 95)
anomaly_labels_train = detect_anomalies_with_threshold(anomaly_scores_train , threshold)
anomaly_labels_test = detect_anomalies_with_threshold(anomaly_scores_test , threshold)