# Breast cancer classification using transfer Learning algorithm#

from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.utils.multiclass import unique_labels
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from sklearn.externals import joblib
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import matplotlib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

#Transfer learning on mammograophic mass dataset which is publicly available
mass = pd.read_csv('/Users/aydafarhadi/Desktop/BreastCancer/UCI/Cleaned_data.csv', header=None)
mass2 = mass[1:]

class TraditionalClassifiers(data):

    def __init__(self):
        self.data=data

    def read_Dataset(df_sas):
        return df_sasr

    df_sasr=read_Dataset(df_sas)

    sc = StandardScaler()
    ## Data loaders
    ## Data loaders
    def load_breast():
        breast_data = imbalancedData
        y = breast_data.iloc[:, 5]
        X = breast_data.iloc[:, :4]
        X = sc.fit_transform(X)
        y = y.values
        return X, y

    X,y=load_breast()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)


    loadModel = load_model('dnn_breast.h5')

    #define AUC for cost function
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    #define mean
    def mean(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)


    # Transfer Learner
    def make_classification(X, y):
        historyauc = []
        historyloss = []

        early_stop_callback = [EarlyStopping(monitor='val_loss',patience=500, verbose=1)
                           ]
        auccores = []
        precisionscores = []
        recallscores = []
        accuscores = []
        seed = 7
        f1scores = []

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        opt = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        # evaluation based on k-fold cross validation
        for train, test in kfold.split(X, y):
            model = Sequential()
            model.add(Dense(100, input_dim=10, kernel_initializer='normal', activation='elu'))
            model.add(Dense(80, kernel_initializer='normal', activation='elu'))
            model.add(Dense(50, kernel_initializer='normal', activation='elu'))
            model.layers[2].set_weights(loadModel.layers[2].get_weights())
            model.layers[2].trainable = True
            model.add(Dense(150, kernel_initializer='normal', activation='elu'))
            model.add(Dropout(0.25))
            model.add(Dense(50, kernel_initializer='normal', activation='elu'))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[auc])
            history = model.fit(X[train], y[train], validation_split=0.3, epochs=1000, batch_size=50, verbose=0,
                             callbacks=early_stop_callback)
            historyauc.append(history.history['val_auc'])
            historyloss.append(history.history['val_loss'])
            predictions = model.predict_classes(X[test])
            cnf_matrix = confusion_matrix(y[test], predictions)
            #print confusion matrix in each iteration
            #print(cnf_matrix)
            precisionscores.append(precision_score(y[test], predictions))
            auccores.append(roc_auc_score(y[test], predictions, average=None))
            recallscores.append(recall_score(y[test], predictions))
            accuscores.append(accuracy_score(y[test], predictions))
            f1scores.append(f1_score(y[test], predictions))
        return precisionscores, auccores, recallscores, accuscores, f1scores


    precisionscores, aucscores, recallscores, accuscores, f1scores = make_classification(X, y)

    #Get evaluation of performances
    def print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores):
        avg_auc = mean(aucscores)
        avg_recall = mean(recallscores)
        avg_precision = mean(precisionscores)
        avg_f1score = mean(f1scores)
        avg_acc = mean(accuscores)
        return (avg_auc, avg_recall, avg_precision, avg_acc, avg_f1score)


    print(print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores))
