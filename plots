import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, missingno as msno, statsmodels.api as sm

def plot_treatments(df_path,docu_path):
    df = pd.read_csv(df_path, sep = ',')
    documentation = pd.read_csv(docu_path, sep = ';')
    select_tutorials = documentation.loc[(documentation['Tutorial'] == "YES")]
    list_tutorials = list(select_tutorials.loc[:,"Variable"])
    freqTable = ['']*4
    treated_table = ['']*4
    total = []
    for i, tutorial in enumerate(list_tutorials):
        freqTable[i] = pd.crosstab(index = df['Ano_Ingreso'], columns = df[tutorial], margins = True)
        freqTable[i].columns = ["Control","Treated","rowTotal"]
        treated_table[i] = freqTable[i]["Treated"]
        total = freqTable[0]["rowTotal"]
    df_treated = pd.DataFrame(pd.concat([treated_table[0],treated_table[1],treated_table[2],treated_table[3]], axis = 1))
    df_treated.columns = list_tutorials
    df_treated["Total"] = total
    for tutorial in list_tutorials:
        df_treated[tutorial] = df_treated[tutorial]/df_treated["Total"]
    del df_treated["Total"]
    df_treated["Year"] = df_treated.index
    df_treated = df_treated[:-1]
    df_treated_plot = pd.melt(df_treated, ['Year'])
    df_treated_plot["value"] = df_treated_plot["value"]*100
    pl_stu = sns.set(style="white")
    pl_stu = sns.barplot(x="Year", y="value", hue="variable", palette= "GnBu_d",data=df_treated_plot)
    pl_stu = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pl_stu = plt.ylabel("Percentage of total students")
    return pl_stu;

def plot_roc(performance):
    pl_stu = sns.set(style="white")
    sns.lineplot(performance[0][0],performance[0][1],label="Fold 1, AUC ="+str(round(performance[0][2],2)))
    sns.lineplot(performance[1][0],performance[1][1],label="Fold 2, AUC ="+str(round(performance[1][2],2)))
    sns.lineplot(performance[2][0],performance[2][1],label="Fold 3, AUC ="+str(round(performance[2][2],2)))
    sns.lineplot(performance[3][0],performance[3][1],label="Fold 4, AUC ="+str(round(performance[3][2],2)))
    sns.lineplot(performance[4][0],performance[4][1],label="Fold 5, AUC ="+str(round(performance[4][2],2)))
    sns.lineplot([0, 1], [0, 1], color="black")
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt;

def plot_obs_u(df,treatment,target,col_period,direction):
    
    def obs_rates(df,treatment,target,col_period,period,direction):
        n_y1_trt = sum(np.where((df[treatment]==1) & (df[target]==1) & (df[col_period]==period),1,0))
        n_y1_ctl = sum(np.where((df[treatment]==0) & (df[target]==1) & (df[col_period]==period),1,0))
        n_trt = sum(np.where((df[treatment]==1) & (df[col_period]==period),1,0))
        n_ctl = sum(np.where((df[treatment]==0) & (df[col_period]==period),1,0))
        y1_r_trt = n_y1_trt/n_trt
        y1_r_ctl = n_y1_ctl/n_ctl
        if direction == 1:
            uplift = round(y1_r_trt - y1_r_ctl,3)
        else:
            uplift = round(y1_r_ctl - y1_r_trt,3)
        data = {'Period': period,'Treatment' : [y1_r_trt],'Control' : [y1_r_ctl],'uplift' : [uplift]}
        df = pd.DataFrame(data,columns = ['Period','Treatment','Control','uplift'])
        return df
    
    list_years = list(df[col_period].unique())
    list_summary = [""]*len(list_years)
    for i in list(range(len(list_years))):
        list_summary[i] = obs_rates(df,treatment,target,col_period,list_years[i],direction)
    df_summary = pd.concat(list_summary)
    df_sum_no_up = pd.melt(df_summary.drop(["uplift"], axis = 1),['Period'])
    #df_sum_no_up.columns = ["Period"]
    df_sum_no_up["Period"] = df_sum_no_up["Period"].astype(str)
    df_summary["Period"] = df_summary["Period"].astype(str)
    sns.set_style("white")
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    sns.barplot(x="Period", y="value", hue="variable", palette= "GnBu_d",data=df_sum_no_up, ax = ax1)
    sns.lineplot(x=list(df_summary['Period']), y=list(df_summary['uplift']),color='r',marker="o",ax=ax2)
    return plt;
