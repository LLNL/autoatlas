from sklearn import linear_model,svm,neighbors,neural_network,metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

def perc_error(y_true,y_pred):
    return 100*np.mean(np.absolute(y_pred-y_true)/y_true)

def reg_scorer(sc_true,sc_rnd):
    return 100*(sc_rnd-sc_true)/sc_true

def class_scorer(sc_true,sc_rnd):
    return -100*(sc_rnd-sc_true)/sc_true

def bacc_scorer(y_true,y_pred):
    return metrics.balanced_accuracy_score(y_true=y_true,y_pred=y_pred)

RLEARN_ARGS = [
    {'task':'regression',
     'estimator':GridSearchCV(estimator=linear_model.Ridge(random_state=0,solver='lsqr'),
                        param_grid={'alpha':[10**N for N in np.arange(-5,6,0.25)]},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'feature_scorers':{'mae':reg_scorer}, 
     'tag':'lin'},

    {'task':'regression',
     'estimator':GridSearchCV(estimator=neighbors.KNeighborsRegressor(),
                        param_grid={'n_neighbors':list(set([int(2**N) for N in np.arange(0,8,0.25)]))},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'feature_scorers':{'mae':reg_scorer}, 
     'tag':'nneigh'},

    {'task':'regression',
     'estimator':GridSearchCV(estimator=svm.SVR(gamma='auto'),
                        param_grid={'C':[10**N for N in np.arange(-5,6,0.25)]},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'feature_scorers':{'mae':reg_scorer}, 
     'tag':'svm'},

    {'task':'regression',
     'estimator':GridSearchCV(estimator=
                neural_network.MLPRegressor(random_state=0,hidden_layer_sizes=(4,2),max_iter=1e5,solver='lbfgs'),
                param_grid={'alpha':[10**N for N in np.arange(-5,6,0.25)]},
                scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                n_jobs=-1,refit=True,cv=3),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'feature_scorers':{'mae':reg_scorer}, 
     'tag':'mlp'},

    {'task':'classification',
     'estimator':GridSearchCV(estimator=
        linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=1e5,multi_class='multinomial'),
        param_grid={'C':[10**N for N in np.arange(-5,6,0.25)]},
        scoring=metrics.make_scorer(metrics.accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3),
     'scorers':{'acc':metrics.accuracy_score,'roc':metrics.roc_auc_score},
     'feature_scorers':{'acc':class_scorer}, 
     'tag':'lin'},

    {'task':'classification',
     'estimator':GridSearchCV(estimator=svm.SVC(gamma='auto',random_state=0),
        param_grid={'C':[10**N for N in np.arange(-5,6,0.25)]},
        scoring=metrics.make_scorer(metrics.accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3),
     'scorers':{'acc':metrics.accuracy_score,'roc':metrics.roc_auc_score},
     'feature_scorers':{'acc':class_scorer}, 
     'tag':'svm'},

    {'task':'classification',
     'estimator':GridSearchCV(estimator=neighbors.KNeighborsClassifier(),
        param_grid={'n_neighbors':list(set([int(2**N) for N in np.arange(0,8,0.25)]))},
        scoring=metrics.make_scorer(metrics.accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3),
     'scorers':{'acc':metrics.accuracy_score,'roc':metrics.roc_auc_score},
     'feature_scorers':{'acc':class_scorer}, 
     'tag':'nneigh'},

    {'task':'classification',
     'estimator':GridSearchCV(estimator=
            neural_network.MLPClassifier(hidden_layer_sizes=(4,2),max_iter=1e5,solver='lbfgs',random_state=0),
            param_grid={'alpha':[10**N for N in np.arange(-5,6,0.25)]},
            scoring=metrics.make_scorer(metrics.accuracy_score,greater_is_better=True),
            n_jobs=-1,refit=True,cv=3),
     'scorers':{'acc':metrics.accuracy_score,'roc':metrics.roc_auc_score},
     'feature_scorers':{'acc':class_scorer}, 
     'tag':'mlp'},
] 
