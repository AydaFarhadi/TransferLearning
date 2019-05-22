# Breast cancer classification using traditional Machien learning algorithms#

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.utils import resample
from sklearn import metrics
import matplotlib.pyplot as plt
import subprocess
from imblearn import under_sampling, over_sampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# Boosting classifiers to handle imbalanced datasets of Mammographic mass dataset
data= pd.read_csv('/Users/aydafarhadi/Desktop/BreastCancer/UCI/Cleaned_data.csv', header=None)
data = data[1:]

class BoostingClassifiers(data):

    def __init__(self):
        self.data=data

    # simulation to make imabalnced classes with different ratio
    negative = data[data[5] == "0"]
    postivie =data[data[5] == "1"]


    # create imbalanced datasets using multiple ratios (2% 378, 5% 353, %10 303)
    IR_List=[378,353,303]

    subsets=[]

    for i in IR_List:
        remove_n = i
        drop_indices = np.random.choice(postivie.index, remove_n, replace=False)
        df_subset = postivie.drop(drop_indices)
        subsets.append(df_subset )


    print("small class size is", subsets.shape)
    print("untouched class size is", negative.shape)

    frames = [negative, df_subset]
    twopercent = pd.concat(frames)

    imbalancedData = twopercent.sort_values([2, 3], ascending=[True, False])

    sc = StandardScaler()


    ## Data loaders
    def load_breast():
        breast_data = imbalancedData
        y = breast_data.iloc[:, 5]
        X = breast_data.iloc[:, :4]
        #X = sc.fit_transform(X)
        y = y.values
        return X, y


    X, y = load_breast()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # ADAboost classifier
    def adaboost(X_train, X_test, y_train):
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


    rus = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42)

    #SMOTE sampling
    def smote_sample(X_train, y_train):
        X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
        y_smote = adaboost(X_train_sm, X_test, y_train_sm)
        return y_smote

    y_smote=smote_sample(X_train, y_train)


    # RUS sampling
    def rus_sample(X_train,y_train):
        X_full = X_train.copy()
        X_full['target'] = y_train
        X_maj = X_full[X_full.target==0]
        X_min = X_full[X_full.target==1]
        X_maj_rus = resample(X_maj,replace=False,n_samples=len(X_min),random_state=44)
        X_rus = pd.concat([X_maj_rus, X_min])
        X_train_rus = X_rus.drop(['target'], axis=1)
        y_train_rus = X_rus.target
        y_rus = adaboost(X_train_rus, X_test, y_train_rus)
        return y_rus

    y_rus=rus_sample(X_train,y_train)


    # Get confusion matrix of model
    cnf_matrix = confusion_matrix(y_test, y_smote)
    print(cnf_matrix)


    # print performances of models
    def print_performance(y_test, y_smote):
        print("precision",precision_score(y_test, y_smote))
        print("AUC score",roc_auc_score(y_test, y_smote))
        print("accuracy",accuracy_score(y_test, y_smote))
        print("f1score",f1_score(y_test, y_smote))
        print("recall",recall_score(y_test, y_smote))



    print_performance(y_test, y_smote)


