import numpy as np
import pandas as pd
from NASA.nasa import NASA

for channel_id in NASA.channel_ids:
    if channel_id == "T-10":
        continue
    print(f"Processing channel: {channel_id}")
    
    # Lista per memorizzare i segmenti di training
    X_train_final = []

    # Uso del dataset NASA per tutti i canali
    dataset = NASA("./datasets", channel_id, mode="anomaly")
    data = dataset.data
    train = []
    for i in range(0, data.shape[0] - STEP +1, OFFSET): 
        train.append(data[i:i+STEP])

    train = np.stack(train)

    # Estrazione anomalie
    dataset = NASA("./datasets", channel_id, mode="anomaly", train=False)
    data = dataset.data
    Test = []
    output = []
    o = np.zeros(data.shape[0])
    for start,end in dataset.anomalies:
        o[start:end] = 1
    for i in range(0, data.shape[0] - STEP +1, OFFSET): 
        Test.append(data[i:i+STEP])
        output.append(o[i:i+STEP])

    output = np.stack(output)
    Test = np.stack(Test)