from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold


# Breast cancer classification using traditional Machine learning algorithms#
'''
This class is udef to implemnt traditional classifiers on Mammographic mass dataset 
using 5 fold cross validation and return all performances used for evaluation of models.
'''

# traditional classification of Mammographic mass dataset
data = pd.read_csv('/Users/aydafarhadi/Desktop/BreastCancer/UCI/Cleaned_data.csv', header=None)
data = data[1:]

class TraditionalClassifiers(data):

    def __init__(self):
        self.data=data

    # simulation to make imabalnced classes with different ratio
    negative = data[data[5] == "0"]
    postivie = data[data[5] == "1"]

    # 2% 378, 5% 353, %10 303
    remove_n = 378
    drop_indices = np.random.choice(postivie.index, remove_n, replace=False)
    df_subset = postivie.drop(drop_indices)

    print("small class size is", df_subset.shape)

    print("untouched class size is", negative.shape)

    frames = [negative, df_subset]
    twopercent = pd.concat(frames)

    imbalancedData = twopercent.sort_values([2, 3], ascending=[True, False])

    # imbalancedData.to_csv("imbalancedData.csv")
    # imbalancedData=pd.read_csv("imbalancedData.csv")

    sc = StandardScaler()


    ## Data loaders
    def load_breast():
        breast_data = imbalancedData
        y = breast_data.iloc[:, 5]
        X = breast_data.iloc[:, :4]
        X = sc.fit_transform(X)
        y = y.values
        return X, y


    X, y = load_breast()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    #Visualize data based on classes
    def plot_classes(data):
        # plot classes
        count_classes = pd.value_counts(data.iloc[:, 5], sort=True)
        count_classes.plot(kind='bar', rot=0)
        plt.title("Mammographic Mass class distribution")
        plt.xlabel("balanced-class")
        plt.ylabel("Frequency");
        pic = plt.savefig('Mammo.png', dpi=80)
        print("value counts are\n", pd.value_counts(y, sort=True))
        return pic


    plot_classes(imbalancedData)

    #get mean of lists
    def mean(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)


    def make_classification(X, y):
        # Logistic regression
        seed = 7
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        auccores = []
        precisionscores = []
        recallscores = []
        accuscores = []
        f1scores = []
        lr = LogisticRegression(C=1.0)

        for train, test in kfold.split(X, y):
            lr.fit(X[train], y[train])
            predictions1 = lr.predict(X[test])
            predictions2 = lr.predict_proba(X[test])[:, 0]
            cnf_matrix = confusion_matrix(y[test], predictions1)
            print(cnf_matrix)

            precisionscores.append(precision_score(y[test], predictions1))
            auccores.append(roc_auc_score(y[test], predictions1, average=None))
            recallscores.append(recall_score(y[test], predictions1))
            accuscores.append(accuracy_score(y[test], predictions1))
            f1scores.append(f1_score(y[test], predictions1))
        return precisionscores, auccores, recallscores, accuscores, f1scores


    precisionscores, aucscores, recallscores, accuscores, f1scores = make_classification(X, y)

    # function to get mean of all performances
    def print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores):
        avg_auc = mean(aucscores)
        avg_recall = mean(recallscores)
        avg_precision = mean(precisionscores)
        avg_f1score = mean(f1scores)
        avg_acc = mean(accuscores)
        return (avg_auc, avg_recall, avg_precision, avg_acc, avg_f1score)

    # print performances
    print(print_performance(precisionscores, aucscores, recallscores, accuscores, f1scores))


    '''
    All other classifiers are defined here, the same function will be 
    used for printing the performance of each classifier
    '''

    # K nearest Neighbor
    seed=7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    knn=KNeighborsClassifier(n_neighbors=3)
    cvscores = []
    auccores = []
    precisionscores=[]
    recallscores=[]
    accuscores=[]
    seed =7
    f1scores=[]

    #evaluation based on k-fold cross validation
    for train, test in kfold.split(X, y):
        knn.fit(X[train],y[train])
        predictions1 = knn.predict(X[test])
        predictions2= knn.predict_proba(X[test])[:, 0]

        cnf_matrix = confusion_matrix(y[test], predictions1)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions1))
        auccores.append(roc_auc_score(y[test], predictions1, average=None))
        recallscores.append(recall_score(y[test], predictions1))
        accuscores.append(accuracy_score(y[test], predictions1))
        f1scores.append(f1_score(y[test], predictions1))


    #GaussianNaiveBayes
    gnb= GaussianNB()
    seed=7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    auccores = []
    precisionscores=[]
    recallscores=[]
    accuscores=[]
    f1scores=[]

    for train, test in kfold.split(X, y):
        gnb.fit(X[train],y[train])
        predictions1 = gnb.predict(X[test])
        predictions2= gnb.predict_proba(X[test])[:, 0]

        cnf_matrix = confusion_matrix(y[test], predictions1)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions1))
        auccores.append(roc_auc_score(y[test], predictions1, average=None))
        recallscores.append(recall_score(y[test], predictions1))
        accuscores.append(accuracy_score(y[test], predictions1))
        f1scores.append(f1_score(y[test], predictions1))


    #RandomForestClassifier
    clf = RandomForestClassifier(class_weight='balanced',n_estimators=300, max_depth=8,random_state=0)

    seed=7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    auccores = []
    precisionscores=[]
    recallscores=[]
    accuscores=[]
    f1scores=[]

    for train, test in kfold.split(X, y):
        clf.fit(X[train],y[train])
        predictions1 = clf.predict(X[test])
        predictions2= clf.predict_proba(X[test])[:, 0]

        cnf_matrix = confusion_matrix(y[test], predictions1)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions1))
        auccores.append(roc_auc_score(y[test], predictions1, average=None))
        recallscores.append(recall_score(y[test], predictions1))
        accuscores.append(accuracy_score(y[test], predictions1))
        f1scores.append(f1_score(y[test], predictions1))


    #Extreme GBoost
    xgb = xgboost.XGBClassifier()
    seed=7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    auccores = []
    precisionscores=[]
    recallscores=[]
    accuscores=[]
    f1scores=[]

    for train, test in kfold.split(X, y):
        xgb.fit(X[train],y[train])
        predictions1 = xgb.predict(X[test])
        predictions2= xgb.predict_proba(X[test])[:, 0]

        cnf_matrix = confusion_matrix(y[test], predictions1)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions1))
        auccores.append(roc_auc_score(y[test], predictions1, average=None))
        recallscores.append(recall_score(y[test], predictions1))
        accuscores.append(accuracy_score(y[test], predictions1))
        f1scores.append(f1_score(y[test], predictions1))


    #MultiLayerPerceptron
    mlp= MLPClassifier()
    seed=7
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    auccores = []
    precisionscores=[]
    recallscores=[]
    accuscores=[]
    f1scores=[]

    for train, test in kfold.split(X, y):
        mlp.fit(X[train],y[train])
        predictions1 = mlp.predict(X[test])
        predictions2= mlp.predict_proba(X[test])[:, 0]

        cnf_matrix = confusion_matrix(y[test], predictions1)
        print(cnf_matrix)
        precisionscores.append(precision_score(y[test], predictions1))
        auccores.append(roc_auc_score(y[test], predictions1, average=None))
        recallscores.append(recall_score(y[test], predictions1))
        accuscores.append(accuracy_score(y[test], predictions1))
        f1scores.append(f1_score(y[test], predictions1))


