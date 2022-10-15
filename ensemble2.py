import efficientnet.tfkeras as efn 
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from sklearn.ensemble import VotingClassifier
from balance import dividirEBalancearPorClasse, ler_BalanceamentoDividido, num_classes_maper
import numpy as np
from tensorflow.keras.applications import MobileNetV2, MobileNet, NASNetMobile, DenseNet121, DenseNet201, Xception, InceptionV3
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
# from skmultilearn.ensemble import voting
from tensorflow import math
# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
from classification import construirClassificador, imprimirResultado
import os

num_classes = 6  # 3 or 6
X_treinamento, y_treinamento, X_validacao, y_validacao, X_teste, y_teste = ler_BalanceamentoDividido(num_classes)
X_treinamento = np.array(X_treinamento, dtype=np.int32)
X_validacao = np.array(X_validacao, dtype=np.int32)
X_teste = np.array(X_teste, dtype=np.int32)
y_treinamento = to_categorical(y_treinamento, num_classes, dtype=np.int32)
y_validacao = to_categorical(y_validacao, num_classes, dtype=np.int32)
y_teste = to_categorical(y_teste, num_classes, dtype=np.int32)

# pasta = '../Modelos ' + str(num_classes) + ' classes/'

# 1 - EfficientNetB0
# 2 - EfficientNetB1
# 3 - EfficientNetB2
# 4 - EfficientNetB3
# 5 - EfficientNetB4
# 6 - EfficientNetB5
# 7 - EfficientNetB6
# 8 - MobileNet
# 9 - Xception
# 10 - InceptionV3

# (2-3-6 classes)
# classifier_1
# - NILM - 1-2-5
# - classifier_2
#   - classifier_3 (Low-grade lesions)
#     - ASC-US - 0-1-1
#     - LSIL   - 0-1-4
#   - classifier_4 (High-grade lesions)
#     - ASC-H - 0-0-0
#     - HSIL  - 0-0-3
#     - SCC   - 0-0-2
classifier_1 = construirClassificador(num_classes_maper(1), 1)
classifier_1.load_weights(os.path.join('models_2', 'best_model_ef0_classifier_1.hdf5'))
y_pred_classifier_1 = classifier_1.predict(X_teste)
y_pred_classifier_1 = np.argmax(y_pred_classifier_1, axis=1)  # 1 = NILM, 0 = Others

X_teste_classifier_2 = X_teste[np.where(y_pred_classifier_1 == 0)]
classifier_2 = construirClassificador(num_classes_maper(2), 1)
classifier_2.load_weights(os.path.join('models_2', 'best_model_ef0_classifier_2.hdf5'))
if len(X_teste_classifier_2) > 0:
    y_pred_classifier_2 = classifier_2.predict(X_teste_classifier_2)
    y_pred_classifier_2 = np.argmax(y_pred_classifier_2, axis=1)  # 1 = ASC-US/LSIL, 0 = ASC-H/HSIL/SCC
else:
    y_pred_classifier_2 = np.array([], dtype='int64')

if num_classes == 3:
    y_pred_classifier_1[np.where(y_pred_classifier_1 == 1)] = 2
    y_pred_classifier_1[np.where(y_pred_classifier_1 == 0)] = y_pred_classifier_2
    y_pred_ensemble = y_pred_classifier_1
else:
    y_pred_classifier_1[np.where(y_pred_classifier_1 == 1)] = 5

    X_teste_classifier_3 = X_teste_classifier_2[np.where(y_pred_classifier_2 == 1)]
    classifier_3 = construirClassificador(num_classes_maper(3), 1)
    classifier_3.load_weights(os.path.join('models_2', 'best_model_ef0_classifier_3_2.hdf5'))
    if len(X_teste_classifier_3) > 0:
        y_pred_classifier_3 = classifier_3.predict(X_teste_classifier_3)
        y_pred_classifier_3 = np.argmax(y_pred_classifier_3, axis=1)  # 0 = ASC-US, 1 = LSIL
        y_pred_classifier_3[np.where(y_pred_classifier_3 == 0)] = 1
        y_pred_classifier_3[np.where(y_pred_classifier_3 == 1)] = 4
    else:
        y_pred_classifier_3 = np.array([], dtype='int64')

    X_teste_classifier_4 = X_teste_classifier_2[np.where(y_pred_classifier_2 == 0)]
    classifier_4 = construirClassificador(num_classes_maper(4), 1)
    classifier_4.load_weights(os.path.join('models_2', 'best_model_ef0_classifier_4.hdf5'))
    if len(X_teste_classifier_4) > 0:
        y_pred_classifier_4 = classifier_4.predict(X_teste_classifier_4)
        y_pred_classifier_4 = np.argmax(y_pred_classifier_4, axis=1)  # 0 = ASC-H, 1 = HSIL, 2 = SCC
        y_pred_classifier_4[np.where(y_pred_classifier_4 == 1)] = 3
    else:
        y_pred_classifier_4 = np.array([], dtype='int64')

    y_pred_classifier_2[np.where(y_pred_classifier_2 == 1)] = y_pred_classifier_3
    y_pred_classifier_2[np.where(y_pred_classifier_2 == 0)] = y_pred_classifier_4
    y_pred_classifier_1[np.where(y_pred_classifier_1 == 0)] = y_pred_classifier_2
    y_pred_ensemble = y_pred_classifier_1

    y_teste = np.argmax(y_teste, axis=1)

    # y_pred_ensemble[np.where(y_teste == 1)] = y_teste[np.where(y_teste == 1)]
    # y_pred_ensemble[np.where(y_teste == 4)] = y_teste[np.where(y_teste == 4)]

    # y_pred_ensemble[np.where(y_teste == 0)] = y_teste[np.where(y_teste == 0)]
    # y_pred_ensemble[np.where(y_teste == 3)] = y_teste[np.where(y_teste == 3)]
    # y_pred_ensemble[np.where(y_teste == 2)] = y_teste[np.where(y_teste == 2)]

print("\n\n ----- Resultado Ensemble -----")
mcm = multilabel_confusion_matrix(y_teste, np.array(y_pred_ensemble))
imprimirResultado(mcm)
