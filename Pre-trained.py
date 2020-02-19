from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras import backend as K
from sklearn.utils.multiclass import unique_labels
from keras.callbacks import EarlyStopping
from imblearn.combine import SMOTETomek
from sklearn.externals import joblib
from sklearn.utils import class_weight
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model, load_model
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import StratifiedKFold


wdbc=pd.read_csv('/Users/aydafarhadi/Desktop/BreastCancer/UCI/wdbc.csv',header=None)
wpbc=pd.read_csv('/Users/aydafarhadi/Desktop/BreastCancer/UCI/wpbc.csv',header=None)

class PreTrainedModel(dataset1,dataset2):

    def __init__(self):
        self.data=data

    #concatenate two data
    def clean_data(dataset1,dataset2):
        dataset1=dataset1.iloc[:,1:]
        dataset2=dataset2.iloc[:,1:]
        frames = [wpbc, wdbc]
        preData = pd.concat(frames)
        preData = preData.replace('?', np.nan)
        preData = preData.fillna(preData.missForest())
        return preData

    # get prepared data
    dataset=clean_data(wdbc,wpbc)
    print(dataset.head())


    sc = StandardScaler()
    ## data loaders
    def load_breast_test():
        #result, 30,31
	    breast_data =dataset
	    y=breast_data.iloc[0:,30]
	    X=breast_data.drop([31], axis=1)
	    X= sc.fit_transform(X)
	    y = y.values
	    return X, y

    #separate predictors and target
    X,y=load_breast_test()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)


    def deep_learning(X,y):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        # early_stopping = EarlyStopping(monitor='val_loss', patience=200)
        auccores = []
        precisionscores = []
        recallscores = []
        accuscores = []
        seed = 7
        f1scores = []
        nadam = optimizers.Nadam(lr=0.00002, clipnorm=1.)

        # input_layer = Input(shape=(input_dim, ))

    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cvscores = []

    # evaluation based on k-fold cross validation
    for train, test in kfold.split(X, y):
        model = Sequential()
        model.add(Dense(1000, input_dim=36, kernel_initializer='normal', activation='elu'))
        model.add(Dense(80, kernel_initializer='normal', activation='elu'))
        model.add(Dropout(0.25))
        model.add(Dense(50, kernel_initializer='normal', activation='elu'))
        model.add(Dropout(0.25))
        model.add(Dense(30, kernel_initializer='normal', activation='elu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
        history = model.fit(X[train], y[train], validation_split=0.3, epochs=1000, batch_size=50, verbose=0,
                             callbacks=[es, mc])
        predictions = model.predict_classes(X[test])
        cnf_matrix = confusion_matrix(y[test], predictions)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions))
        auccores.append(roc_auc_score(y[test], predictions, average=None))
        recallscores.append(recall_score(y[test], predictions))
        accuscores.append(accuracy_score(y[test], predictions))
        f1scores.append(f1_score(y[test], predictions))
        pyplot.plot(history.history['acc'], label='train')
        pyplot.plot(history.history['val_acc'], label='validation')
        pyplot.legend()
        #pyplot.show()
        pic = pyplot.savefig('AUC.png', dpi=80)
    return precisionscores, auccores, recallscores, accuscores, f1scores,history,model3


    precisionscores, aucscores, recallscores, accuscores, f1scores,history,pre-trained-model=deep_learning(X,y)

    #calculate mean of numbers
    def mean(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)

    #Get all performances
    def print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores):
        avg_auc = mean(aucscores)
        avg_recall = mean(recallscores)
        avg_precision = mean(precisionscores)
        avg_f1score = mean(f1scores)
        avg_acc = mean(accuscores)
        return (avg_auc, avg_recall, avg_precision, avg_acc, avg_f1score)


    print(print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores))

    #save pre-trained model
    #plot_model(model3, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    pre-trained-model.save('dnn_breast.h5')
    #loadedModel = load_model('dnn_breast.h5')
    #print(loadedModel.summary())
