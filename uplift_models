import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, missingno as msno, statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def uplift_model(df_train,df_test,treatment,predictors,target, direction, model):
    random_state = 100
    if model == "SMARF":
        np.random.seed(random_state)
        dfTreatment = df_train[df_train[treatment] == 1]
        dfControl = df_train[df_train[treatment] == 0]
        crf_treatment = RandomForestClassifier(n_estimators = 500,random_state = random_state)
        crf_control = RandomForestClassifier(n_estimators = 500,random_state = random_state)
        model_Treatment = crf_treatment.fit(dfTreatment[predictors],dfTreatment[target])
        model_Control = crf_control.fit(dfControl[predictors],dfControl[target])
        testIndices = df_test[predictors].index.values
        prob_treat = model_Treatment.predict_proba(df_test[predictors])[:,1]
        prob_ctrl = model_Control.predict_proba(df_test[predictors])[:,1]
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
    
    elif model == "SMAxgboost":
        np.random.seed(random_state)
        dfTreatment = df_train[df_train[treatment] == 1]
        dfControl = df_train[df_train[treatment] == 0]
        crf_treatment = xgb.XGBClassifier(objective = "binary:logistic",colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 30,random_state = random_state)
        crf_control = xgb.XGBClassifier(objective = "binary:logistic",colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 30,random_state = random_state)
        model_Treatment = crf_treatment.fit(dfTreatment[predictors],dfTreatment[target])
        model_Control = crf_control.fit(dfControl[predictors],dfControl[target])
        testIndices = df_test[predictors].index.values
        prob_treat = model_Treatment.predict_proba(df_test[predictors])[:,1]
        prob_ctrl = model_Control.predict_proba(df_test[predictors])[:,1]
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
    
    elif model == "MOAxgboost":
        np.random.seed(random_state)
        df_train["tr"] = np.where((df_train[treatment] == 1) & (df_train[target] == 1),1,0)
        df_train["cn"] = np.where((df_train[treatment] == 0) & (df_train[target] == 0),1,0)
        df_train["cr"] = np.where((df_train[treatment] == 0) & (df_train[target] == 1),1,0)
        df_train["tn"] = np.where((df_train[treatment] == 1) & (df_train[target] == 0),1,0)
        df_train["new_target"] = np.where((df_train["tr"] == 1) | (df_train["cn"] == 1),1,0)
        xg_model = xgb.XGBClassifier(objective = "binary:logistic",colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 30,random_state = random_state)
        fitted_model = xg_model.fit(df_train[predictors],df_train["new_target"])
        testIndices = df_test[predictors].index.values
        prob_treat = fitted_model.predict_proba(df_test[predictors])[:,1]
        prob_ctrl = fitted_model.predict_proba(df_test[predictors])[:,0]
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
        #feature_importances = pd.DataFrame(fitted_model.feature_importances_,
                                          #index = predictors,
                                          #columns = ['importance']).sort_values('importance',ascending =False)
        
        
    elif model == "MOARF":
        np.random.seed(random_state)
        df_train["tr"] = np.where((df_train[treatment] == 1) & (df_train[target] == 1),1,0)
        df_train["cn"] = np.where((df_train[treatment] == 0) & (df_train[target] == 0),1,0)
        df_train["cr"] = np.where((df_train[treatment] == 0) & (df_train[target] == 1),1,0)
        df_train["tn"] = np.where((df_train[treatment] == 1) & (df_train[target] == 0),1,0)
        df_train["new_target"] = np.where((df_train["tr"] == 1) | (df_train["cn"] == 1),1,0)
        rf_model = RandomForestClassifier(n_estimators = 500,random_state = random_state)
        fitted_model = rf_model.fit(df_train[predictors],df_train["new_target"])
        testIndices = df_test[predictors].index.values
        prob_treat = fitted_model.predict_proba(df_test[predictors])[:,1]
        prob_ctrl = fitted_model.predict_proba(df_test[predictors])[:,0]
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
        #feature_importances = pd.DataFrame(fitted_model.feature_importances_,
                                          #index = predictors,
                                          #columns = ['importance']).sort_values('importance',ascending =False)
    
    elif model == "MCARF":
        np.random.seed(random_state)
        xt = df_train[predictors].multiply(df_train[treatment], axis = "index")
        xt = xt.add_suffix('_inter')
        predictors_and_interactions = pd.concat([df_train[predictors],xt,df_train[treatment]], axis = 1)
        list_pred_int = list(predictors_and_interactions.columns)
        df_train = pd.concat([df_train[target],predictors_and_interactions],axis = 1)
        rf_model = RandomForestClassifier(n_estimators = 500,random_state = random_state)
        fitted_model = rf_model.fit(df_train[list_pred_int],df_train[target])
        testIndices = df_test[predictors].index.values
        # CT 
        df_ct = df_test.copy()
        xt_c = df_ct[predictors].multiply(0, axis = "index")
        xt_c = xt_c.add_suffix('_inter')
        df_ct[treatment] = 0
        pred_int_ct = pd.concat([df_ct[predictors],df_ct[treatment],xt_c],axis = 1)
        list_pred_int_ct = list(pred_int_ct.columns)
        prob_ctrl = fitted_model.predict_proba(pred_int_ct)[:,1]
        # TR
        df_tr = df_test.copy()
        xt_tr = df_tr[predictors].multiply(1, axis = "index")
        xt_tr = xt_tr.add_suffix('_inter')
        df_tr[treatment] = 1
        pred_int_tr = pd.concat([df_tr[predictors],df_tr[treatment],xt_tr],axis = 1)
        list_pred_int_tr = list(pred_int_tr.columns)
        prob_treat = fitted_model.predict_proba(pred_int_tr)[:,1]
        # Summary
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
        #feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                          #index = list_pred_int,
                                          #columns = ['importance']).sort_values('importance',ascending =False)
        
    elif model == "MCAxgboost":
        np.random.seed(random_state)
        xt = df_train[predictors].multiply(df_train[treatment], axis = "index")
        xt = xt.add_suffix('_inter')
        predictors_and_interactions = pd.concat([df_train[predictors],xt,df_train[treatment]], axis = 1)
        list_pred_int = list(predictors_and_interactions.columns)
        df_train = pd.concat([df_train[target],predictors_and_interactions],axis = 1)
        xg_model = xgb.XGBClassifier(objective = "binary:logistic",colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 30,random_state = random_state)
        fitted_model = xg_model.fit(df_train[list_pred_int],df_train[target])
        testIndices = df_test[predictors].index.values
        # CT 
        df_ct = df_test.copy()
        xt_c = df_ct[predictors].multiply(0, axis = "index")
        xt_c = xt_c.add_suffix('_inter')
        df_ct[treatment] = 0
        pred_int_ct = pd.concat([df_ct[predictors],xt_c,df_ct[treatment]],axis = 1)
        list_pred_int_ct = list(pred_int_ct.columns)
        prob_ctrl = fitted_model.predict_proba(pred_int_ct)[:,1]
        # TR
        df_tr = df_test.copy()
        xt_tr = df_tr[predictors].multiply(1, axis = "index")
        xt_tr = xt_tr.add_suffix('_inter')
        df_tr[treatment] = 1
        pred_int_tr = pd.concat([df_tr[predictors],xt_tr,df_tr[treatment]],axis = 1)
        list_pred_int_tr = list(pred_int_tr.columns)
        prob_treat = fitted_model.predict_proba(pred_int_tr)[:,1]
        # Summary
        predictions_treat = pd.DataFrame(prob_treat,testIndices, columns = {"Prob_treat"})
        predictions_ctrl = pd.DataFrame(prob_ctrl,testIndices, columns = {"Prob_ctrl"})
        predictions = pd.concat([df_test,predictions_treat,predictions_ctrl],axis = 1)
        #feature_importances = pd.DataFrame(fitted_model.feature_importances_,
                                           #index = list_pred_int,
                                           #columns = ['importance']).sort_values('importance',ascending =False)
    
    if direction == 1:
        predictions["s"] = predictions["Prob_treat"] - predictions["Prob_ctrl"]
    elif direction == 2:
        predictions["s"] = predictions["Prob_ctrl"] - predictions["Prob_treat"]
    
    return predictions;

def performance_uplift(predictions,treatment,target,continuous,direction,model_name):
    
    if continuous == True:
        pred_sorted = predictions.sort_values(by = ["s"], ascending = False)
        pred_sorted["n_t"] = np.where(pred_sorted[treatment] == 1,1,0)
        pred_sorted["n_c"] = np.where(pred_sorted[treatment] == 1,0,1)
        pred_sorted["n_y1_t"] = np.where((pred_sorted[treatment] == 1) & (pred_sorted[target] == 1),1,0)
        pred_sorted["n_y1_c"] = np.where((pred_sorted[treatment] == 0) & (pred_sorted[target] == 1),1,0)
        pred_sorted["r_y1_t"] = pred_sorted["n_y1_t"]/ pred_sorted["n_t"]
        pred_sorted["r_y1_c"] = pred_sorted["n_y1_c"]/ pred_sorted["n_c"]
        pred_sorted.fillna(0, inplace = True)
        
        if direction == 1:
            pred_sorted["uplift"] = pred_sorted["r_y1_t"] - pred_sorted["r_y1_c"]
        elif direction == 2:
            pred_sorted["uplift"] = pred_sorted["r_y1_c"] - pred_sorted["r_y1_t"]
        
        performance_elements = ["n_t","n_c","n_y1_t","n_y1_c","r_y1_t","r_y1_c","uplift"]
        performance_table = pred_sorted.loc[:,performance_elements]
        performance_table["r_cum_y1_t"] = np.cumsum(performance_table["n_y1_t"]) / np.cumsum(performance_table["n_t"])
        performance_table["r_cum_y1_c"] = np.cumsum(performance_table["n_y1_c"]) / np.cumsum(performance_table["n_c"])
        performance_table.fillna(0, inplace = True)
    
        if direction == 1:
            inc_gains = (performance_table["r_cum_y1_t"] - performance_table["r_cum_y1_c"]) * ((np.cumsum(performance_table["n_t"]) + np.cumsum(performance_table["n_c"]))/ (np.sum(performance_table["n_t"]) + np.sum(performance_table["n_c"])))
            inc_gains = pd.Series(inc_gains).values
            inc_gains = np.append([0],inc_gains)
            overall_inc_gains = np.sum(performance_table["n_y1_t"])/np.sum(performance_table["n_t"]) - np.sum(performance_table["n_y1_c"])/np.sum(performance_table["n_c"])
            random_inc_gains = np.append([0],np.cumsum(np.repeat(overall_inc_gains / performance_table.shape[0], performance_table.shape[0])))
        elif direction == 2:
            inc_gains = (performance_table["r_cum_y1_c"] - performance_table["r_cum_y1_t"]) * ((np.cumsum(performance_table["n_t"]) + np.cumsum(performance_table["n_c"]))/ (np.sum(performance_table["n_t"]) + np.sum(performance_table["n_c"])))
            inc_gains = pd.Series(inc_gains).values
            inc_gains = np.append([0],inc_gains)
            overall_inc_gains = np.sum(performance_table["n_y1_c"])/np.sum(performance_table["n_c"]) - np.sum(performance_table["n_y1_t"])/np.sum(performance_table["n_t"])
            random_inc_gains = np.append([0],np.cumsum(np.repeat(overall_inc_gains / performance_table.shape[0], performance_table.shape[0])))
        
        x = np.append([0],np.arange(100/performance_table.shape[0],100,100/performance_table.shape[0]))
        x = np.append(x,[100])
        df_inc_gains = pd.DataFrame(inc_gains*100, columns = [model_name])
        df_random_gains = pd.DataFrame(random_inc_gains*100, columns = ["Random"])
        x = pd.DataFrame(x,columns = ["x"])
        df_uplift_curve = pd.concat([x, df_inc_gains,df_random_gains], axis = 1)
        results = [df_uplift_curve,pred_sorted]
    
    elif continuous == False:
        pred_sorted = predictions.sort_values(by = ["s"], ascending = False)
        pred_sorted["ranking"] =  list(range(1,len(pred_sorted)+1))
        labels = [1,2,3,4,5,6,7,8,9,10]
        pred_sorted["bin"] = pd.cut(pred_sorted['ranking'], 10, labels = labels)
        pred_sorted["ct"] = np.where(pred_sorted[treatment]==0,1,0)
        pred_sorted["r_t"] = np.where((pred_sorted[treatment]==1) & (pred_sorted[target]==1),1,0)
        pred_sorted["r_c"] = np.where((pred_sorted[treatment]==0) & (pred_sorted[target]==1),1,0)
        n_c = pred_sorted.pivot_table(index='bin', values= "ct", aggfunc='sum')
        n_t = pred_sorted.pivot_table(index='bin', values= treatment, aggfunc='sum')
        n_y1_c = pred_sorted.pivot_table(index='bin', values= "r_c", aggfunc='sum')
        n_y1_t = pred_sorted.pivot_table(index='bin', values= "r_t", aggfunc='sum')
        perf_table = pd.concat([n_c,n_t,n_y1_c,n_y1_t], axis = 1)
        perf_table.columns = ["n_c","n_t","n_y1_c","n_y1_t"]
        perf_table["r_y1_c"] = perf_table["n_y1_c"]/perf_table["n_c"]
        perf_table["r_y1_t"] = perf_table["n_y1_t"]/perf_table["n_t"]
        
        if direction == 1:
            perf_table["uplift"] = perf_table["r_y1_t"] - perf_table["r_y1_c"]
        elif direction == 2:
            perf_table["uplift"] = perf_table["r_y1_c"] - perf_table["r_y1_t"]
        
        r_cum_y1_c = np.cumsum(perf_table["n_y1_c"])/np.cumsum(perf_table["n_c"])
        r_cum_y1_t = np.cumsum(perf_table["n_y1_t"])/np.cumsum(perf_table["n_t"])
        
        if direction == 1:
            inc_gains = (r_cum_y1_t - r_cum_y1_c)*((np.cumsum(perf_table["n_t"]) + np.cumsum(perf_table["n_c"]))/(np.sum(perf_table["n_t"]) + np.sum(perf_table["n_c"])))
            inc_gains = pd.Series(inc_gains).values
            inc_gains = np.append([0],inc_gains)
            overall_inc_gains = (np.sum(perf_table["n_y1_t"])/np.sum(perf_table["n_t"])) - (np.sum(perf_table["n_y1_c"])/np.sum(perf_table["n_c"]))
            random_inc_gains = np.append([0],np.cumsum(np.repeat(overall_inc_gains / perf_table.shape[0], perf_table.shape[0])))
            
        elif direction == 2:
            inc_gains = (r_cum_y1_c -r_cum_y1_t)*((np.cumsum(perf_table["n_t"]) + np.cumsum(perf_table["n_c"]))/(np.sum(perf_table["n_t"]) + np.sum(perf_table["n_c"])))
            inc_gains = pd.Series(inc_gains).values
            inc_gains = np.append([0],inc_gains)
            overall_inc_gains = (np.sum(perf_table["n_y1_c"])/np.sum(perf_table["n_c"])) - (np.sum(perf_table["n_y1_t"])/np.sum(perf_table["n_t"]))
            random_inc_gains = np.append([0],np.cumsum(np.repeat(overall_inc_gains / perf_table.shape[0], perf_table.shape[0])))
        
        df_inc_gains = pd.DataFrame(inc_gains*100, columns = [model_name])
        df_random_gains = pd.DataFrame(random_inc_gains*100, columns = ["Random"])
        df_uplift_curve = pd.concat([df_inc_gains,df_random_gains], axis = 1)
        df_uplift_curve.index = list(range(0,len(df_uplift_curve)))
        df_uplift_curve["x"] = df_uplift_curve.index*10

        results = [df_uplift_curve,pred_sorted]
    return results;

def qini(perf_object,model_name,folds):
    x = list(perf_object["x"]/100)
    y_inc = list(perf_object[model_name])
    y_ran = list(perf_object["Random"])
    
    def auc(x,y):
        auc = 0
        for i in list(range(1,len(x))):
            auc = auc + 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
        return auc
 
    auc_inc = auc(x,y_inc)
    auc_ran = auc(x,y_ran)
    
    qini = auc_inc - auc_ran
    
    return qini;

def qini_spe(perf_object,percentage,model_name):
    groups = len(perf_object)-1
    deciles = int(percentage*groups/100)
    x = list(perf_object["x"]/100)[0:(deciles+1)]
    y_inc = list(perf_object[model_name][0:(deciles + 1)])
    y_ran = list(perf_object["Random"][0:(deciles + 1)])
    
    def auc(x,y):
        auc = 0
        for i in list(range(1,len(x))):
            auc = auc + 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
        return auc
    
    auc_inc = auc(x,y_inc)
    auc_ran = auc(x,y_ran)
    
    qini = auc_inc - auc_ran
    
    return qini;

def qini_lists_fold(perf_list_model,model_name,amountofFolds):
    def qini_at_fold(perf_list_model,model_name):
        list_qini_at_fold =[]
        for i in list(np.linspace(0, 100, 11, endpoint = True)):
            list_qini_at_fold.append(qini_spe(perf_list_model,i,model_name))
        model_name = pd.DataFrame(list_qini_at_fold, columns = [model_name])
        model_name["x"] = list(np.linspace(0, 100, 11, endpoint = True))
        return model_name;
    qini_folds = []
    for i in list(range(0,amountofFolds)):
        qini_folds.append(qini_at_fold(perf_list_model[i],model_name))
    return qini_folds;
