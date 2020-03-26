from sklearn import linear_model,svm,neighbors,neural_network,metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

def perc_error(y_true,y_pred):
    return 100*np.mean(np.absolute(y_pred-y_true)/y_true)

def reg_scorer(sc_true,sc_rnd):
    return 100*(sc_rnd-sc_true)/sc_true

RLEARN_ARGS = [
    {'type':'regression',
     'estimator':GridSearchCV(estimator=linear_model.Ridge(random_state=0,solver='lsqr'),
                        param_grid={'alpha':[10**N for N in range(-5,6)]},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'region_scorers':{'mae':reg_scorer}, 
     'tag':'lin'},

    {'type':'regression',
     'estimator':GridSearchCV(estimator=svm.SVR(gamma='scale'),
                        param_grid={'C':[10**N for N in range(-5,6)]},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'region_scorers':{'mae':reg_scorer}, 
     'tag':'svm'},

    {'type':'regression',
     'estimator':GridSearchCV(estimator=neighbors.KNeighborsRegressor(),
                        param_grid={'n_neighbors':[2**N for N in range(8)]},
                        scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'region_scorers':{'mae':reg_scorer}, 
     'tag':'nneigh'},

    {'type':'regression',
     'estimator':GridSearchCV(estimator=
                neural_network.MLPRegressor(hidden_layer_sizes=(4,2),max_iter=1e4,solver='lbfgs'),
                param_grid={'alpha':[10**N for N in range(-5,6)]},
                scoring=metrics.make_scorer(metrics.r2_score,greater_is_better=True),
                n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'r2':metrics.r2_score,'mae':metrics.mean_absolute_error,'mape':perc_error},
     'region_scorers':{'mae':reg_scorer}, 
     'tag':'mlp'},

    {'type':'classification',
     'estimator':GridSearchCV(estimator=
        linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=1e4,multi_class='multinomial'),
        param_grid={'C':[10**N for N in range(-5,6)]},
        scoring=metrics.make_scorer(metrics.balanced_accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'bacc':metrics.balanced_accuracy_score,'acc':metrics.accuracy_score},
     'tag':'lin'},

    {'type':'classification',
     'estimator':GridSearchCV(estimator=svm.SVC(gamma='scale',random_state=0),
        param_grid={'C':[10**N for N in range(-5,6)]},
        scoring=metrics.make_scorer(metrics.balanced_accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'bacc':metrics.balanced_accuracy_score,'acc':metrics.accuracy_score},
     'tag':'svm'},

    {'type':'classification',
     'estimator':GridSearchCV(estimator=neighbors.KNeighborsClassifier(),
        param_grid={'n_neighbors':[2**N for N in range(8)]},
        scoring=metrics.make_scorer(metrics.balanced_accuracy_score,greater_is_better=True),
        n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'bacc':metrics.balanced_accuracy_score,'acc':metrics.accuracy_score},
     'tag':'nneigh'},

    {'type':'classification',
     'estimator':GridSearchCV(estimator=
            neural_network.MLPClassifier(hidden_layer_sizes=(4,2),max_iter=1e4,solver='lbfgs',random_state=0),
            param_grid={'alpha':[10**N for N in range(-5,6)]},
            scoring=metrics.make_scorer(metrics.balanced_accuracy_score,greater_is_better=True),
            n_jobs=-1,refit=True,cv=3,iid=False),
     'scorers':{'bacc':metrics.balanced_accuracy_score,'acc':metrics.accuracy_score},
     'tag':'mlp'},
] 
