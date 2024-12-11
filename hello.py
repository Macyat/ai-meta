def add(x, y):
    return x + y


# var =
x1 = 1
y1 = 2
print("This is the sum: 1, 2, %s" % add(1, 2))


def elastic_meta(in_data, target, bound, days, configs, label):
    """
    To train a Linear regression model with combined L1 and L2 priors as regularizer.
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        days_train = days[train_idx]
        parametersGrid = {
            "max_iter": [1, 5],
            "alpha": [0.00000000001, 0.00000001],
            "l1_ratio": [0],
        }
        eNet = ElasticNet()
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X_train, y_train, days[train_idx])
        logo.get_n_splits(groups=days[train_idx])
        # print(len(train_idx))
        # print(len(days[train_idx]))
        clf = GridSearchCV(eNet, parametersGrid, scoring="accuracy", cv=logo)

        # X_train0, y_train = resample_meta(np.concatenate((X_train,days[train_idx].reshape(-1,1)),axis=1), y_train, data_type[train_idx])
        # X_train = X_train0[:,:-1]
        # days_train = X_train0[:,-1]

        # clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)

        # clf = ElasticNet(alpha=configs[label + "2"].values[0], fit_intercept=True)
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train, groups=days_train)
        print("LEN coef", clf.best_params_)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        # if clf.score(X_train, y_train) > best_score1:
        if clf.best_score_ > best_score1:
            # best_score1 = clf.score(X_train, y_train)
            best_score1 = clf.best_score_
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict
