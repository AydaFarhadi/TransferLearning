
class GridSearch(data):

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

    def create_model(learning_rate='0.01', activation='relu',optimizer='adam'):
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
    return model

    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adam','Nadam']
    learning_rate = ['0.01', '0.001', '0.0001']
    activation = ['tanh','lrelu','elu','selu']

    param_grid = dict(optimizer=optimizer, learning_rate, activation )
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
