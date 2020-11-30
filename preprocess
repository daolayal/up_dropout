import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, missingno as msno, statsmodels.api as sm
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def pre_process(df_path,docu_path,name_target,treatment):
    df = pd.read_csv(df_path, sep = ',')
    documentation = pd.read_csv(docu_path, sep = ';')
    pred_rename = documentation[documentation["NewName"].notnull()]
    list_pred_rename = list(pred_rename.loc[:,"Variable"])
    for predictor in list_pred_rename:
        df.rename(columns = {predictor:list(documentation.loc[documentation['Variable'] == predictor,"NewName"])[0]},inplace = True)
    list_predictors = list(documentation.loc[(documentation['Predictor'] == "YES")]["NewName"])
    list_cat_pred = list(df[list_predictors].dtypes.loc[df[list_predictors].dtypes == 'object'].index)
    list_dummy_predictors = ['']*7
    for i,predictor in enumerate(list_cat_pred):
        list_dummy_predictors[i] = pd.get_dummies(df[predictor])
    def new_name_lev(list_levels,list_cat_pred):
        levels_new = ['']*len(list_levels)
        for i,level in enumerate(list_levels):
            levels_new[i] = list_cat_pred + "_" + level[:2] + level[12:13]
        return levels_new;
    levels = ['']*len(list_dummy_predictors)
    for i,predictor in enumerate(list_dummy_predictors):
        levels[i] = list(list_dummy_predictors[i].columns)
        list_dummy_predictors[i].columns = new_name_lev(levels[i],list_cat_pred[i])
    df_dummies = pd.concat(list_dummy_predictors, axis=1)
    list_names_dum = list(df_dummies.columns)
    predictors = [x for x in list_predictors if x not in list_cat_pred]
    predictors = predictors + list_names_dum
    list_float_pred = list(df[list_predictors].dtypes.loc[df[list_predictors].dtypes == 'float64'].index)
    scale_also = ["V17","V29","V41"]
    list_float_pred = list_float_pred + scale_also
    df_float = pd.DataFrame(preprocessing.scale(df[list_float_pred]), columns = df[list_float_pred].columns)
    target = df[name_target]
    treatment = "all"
    if treatment == "all":
        df["treatment"] = np.where((df["V44"] == 1) | (df["V45"] == 1) | (df["V46"] == 1) | (df["V47"] == 1),1,0)
    elif treatment == "mat":
        df["treatment"] = np.where((df["V44"] == 1) & (df["V45"] == 0) & (df["V46"] == 0) & (df["V47"] == 0),1,0)
    elif treatment == "econo":
        df["treatment"] = np.where((df["V44"] == 0) & (df["V45"] == 1) & (df["V46"] == 0) & (df["V47"] == 0),1,0)
    list_dfs_predictors = [df_dummies,df_float]
    df_predictors = pd.concat(list_dfs_predictors, axis=1)
    list_predictors = list(df_predictors.columns)
    list_all_dfs = [df_predictors,target,df["treatment"],df["V1"],df["V2"]]
    df = pd.concat(list_all_dfs, axis=1)
    return [df,list_predictors];

def near_zero_variance(df,predictors):
    threshold = 0
    sel = VarianceThreshold(threshold)
    df_predictors = df[predictors]
    sel.fit(df_predictors)
    idx = np.where(sel.variances_ > threshold)[0]
    non_zero_predictors = list(df_predictors.iloc[:,list(idx)].columns)
    return non_zero_predictors; 

def crossvalidation(df,predictors,target):
    skf = StratifiedKFold(n_splits=5, random_state = 100)
    skf.indeces = list(skf.split(df[predictors],df[target]))
    training = ['']*len(skf.indeces)
    for i in range(len(skf.indeces)):
        training[i] = df.iloc[skf.indeces[i][0]]
    testing = ['']*len(skf.indeces)
    for i in range(len(skf.indeces)):
        testing[i] = df.iloc[skf.indeces[i][1]]
    return [training,testing];

def imba_cor(list_dfs_train,predictors,target):
    os_predictors = ['']*len(list_dfs_train)
    os_target = ['']*len(list_dfs_train)
    os_train = ['']*len(list_dfs_train)
    for i in list(range(len(list_dfs_train))):
        os = SMOTE(random_state = 100)
        os_predictors[i], os_target[i] = os.fit_sample(list_dfs_train[i][predictors],list_dfs_train[i][target])
        os_train[i] = pd.concat([pd.DataFrame(data = os_predictors[i], columns = predictors),pd.DataFrame(data = os_target[i], columns = [target])], axis = 1)
    return os_train;

def apply_woe(df,outcome,list_variables):
    
    def char_bin(Y,X):
        df1 = pd.DataFrame({"X": X, "Y": Y})
        df2 = df1.groupby('X',as_index=True)
        d3 = pd.DataFrame({},index=[])
        d3["COUNT"] = df2.count().Y
        d3["MIN_VALUE"] = df2.sum().Y.index
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        d3["EVENT"] = df2.sum().Y
        d3["NONEVENT"] = df2.count().Y - df2.sum().Y
        d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE']] 
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3 = d3.reset_index(drop=True)
        return d3;
    
    df_woe = pd.DataFrame([])
    for var in list_variables:
        woe_var = char_bin(outcome, df[var])
        woe_var["VAR_NAME"] = var   
        df_woe = df_woe.append(woe_var)

    for var in list_variables:
        small_df = df_woe[df_woe['VAR_NAME'] == var]
        transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))
        replace_cmd = ''
        replace_cmd1 = ''
        for i in sorted(transform_dict.items()):
            replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
            replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
        replace_cmd = replace_cmd + '0'
        replace_cmd1 = replace_cmd1 + '0'
        if replace_cmd != '0':
            try:
                df[var] = df[var].apply(lambda x: eval(replace_cmd))
            except:
                df[var] = df[var].apply(lambda x: eval(replace_cmd1))
    
    return df;

def preprocess_with_woe(df_path,docu_path,name_target,name_treatment):
    
    df = pd.read_csv(df_path, sep = ',')
    documentation = pd.read_csv(docu_path, sep = ';')
    for predictor in list(documentation["Variable"]):
        df.rename(columns = {predictor:list(documentation.loc[documentation['Variable'] == predictor,"NewName"])[0]},inplace = True)
    df = apply_woe(df,df[name_target],list(documentation.loc[(documentation['Type'] == 'object')]['NewName']))
    df["treatment"] = np.where((df["V9"] == 1) | (df["V10"] == 1) | (df["V11"] == 1) | (df["V16"] == 1),1,0)
    int_z = list(documentation.loc[(documentation['Predictor'] == 1) & (documentation['Type'] == 'int64') & (documentation['Unique'] > 2)]['NewName'])
    float_z = list(documentation.loc[(documentation['Predictor'] == 1) & (documentation['Type'] == 'float64')]['NewName'])
    float_z.extend(int_z)
    scaler = StandardScaler()
    df[float_z] = scaler.fit_transform(df[float_z])
    data = {}
    data['df'] = df
    data['predictors'] = list(documentation.loc[(documentation['Predictor'] == 1)]['NewName'])
    data['target'] = name_target
    data['treatment'] = name_treatment
    
    return data;
