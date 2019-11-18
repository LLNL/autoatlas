
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

def train_inf(inf_type,opt,train_input,train_output,test_input,train_mask=None,test_mask=None,test_output=None):
    scaler = StandardScaler().fit(train_input)
    train_input = scaler.transform(train_input) 
    test_input = scaler.transform(test_input) 
 
    reg_params = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4,1e5]
    scorer = 'balanced_accuracy' if inf_type == 'classifier' else 'r2'
    if opt == 'lin':
        if inf_type == 'classifier':
            param_grid = dict(C=reg_params)
            estor = LogisticRegression(random_state=0,solver='lbfgs',max_iter=10000,multi_class='multinomial')
        else:
            param_grid = dict(alpha=reg_params)
            estor = Ridge(random_state=0,solver='lsqr')
    elif opt == 'svm':
        if inf_type == 'classifier':
            param_grid = dict(C=reg_params)
            estor = SVC(gamma='scale',random_state=0)
        else:
            param_grid = dict(C=reg_params)
            estor = SVR(gamma='scale')
    elif opt == 'nneigh':
        param_grid = dict(n_neighbors=[2,4,8,16,32,64,128])
        if inf_type == 'classifier':
            estor = KNeighborsClassifier()
        else:
            estor = KNeighborsRegressor()
    elif opt == 'mlp':
        param_grid = dict(alpha=reg_params) 
        if inf_type == 'classifier':    
            estor = MLPClassifier(hidden_layer_sizes=(4,2),max_iter=10000,solver='lbfgs',random_state=0)
        else:
            estor = MLPRegressor(hidden_layer_sizes=(4,2),max_iter=10000,solver='lbfgs',random_state=0)

    gridestor = GridSearchCV(estimator=estor,param_grid=param_grid,n_jobs=-1,refit=True,cv=3,scoring=scorer,iid=False).fit(train_input[train_mask],train_output) 
    train_pred = gridestor.predict(train_input[train_mask])
    test_pred = gridestor.predict(test_input[test_mask])
    
    best_params = gridestor.best_params_
    print('Task {}: Opt {}: Best params is {}'.format(inf_type,opt,best_params))
    print(gridestor.score(train_input[train_mask],train_output),gridestor.score(test_input[test_mask],test_output))
 
    return train_pred,test_pred,best_params              
            
#print(gridestor.score(train_input[train_mask],train_output),gridestor.score(test_input[test_mask],test_output))
#if inf_type == 'classifier':
#    print(balanced_accuracy_score(gridestor.predict(train_input[train_mask]),train_output),balanced_accuracy_score(gridestor.predict(test_input[test_mask]),test_output))
#classifier = GradientBoostingClassifier().fit(train_input[train_mask],train_output)
#classifier = GradientBoostingRegressor().fit(train_input[train_mask],train_output)
