import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, missingno as msno, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class Dict(object):
    def __getitem__(self, key):
        return self._dict[key]
    def __iter__(self):
        return iter(self._dict)
    def __repr__(self):
        return self._dict.__repr__()
    def keys(self):
        return self._dict.keys()
    def iteritems(self):
        return self._dict.iteritems()
    def get(self, key, default=None):
        return self._dict.get(key, default)

def preprocess(Y, D, X):
    if Y.shape[0] == D.shape[0] == X.shape[0]:
        N = Y.shape[0]
    else:
        raise IndexError('Input data have different number of rows')
    if Y.shape != (N, ):
        Y.shape = (N, )
    if D.shape != (N, ):
        D.shape = (N, )
    if D.dtype != 'int':
        D = D.astype(int)
    if X.shape == (N, ):
        X.shape = (N, 1)
    return (Y, D, X)

def calc_ndiff(mean_c, mean_t, sd_c, sd_t, direction):
    if direction == 1:
        diff = (mean_t-mean_c) / np.sqrt((sd_c**2+sd_t**2)/2)
    elif direction == 2:
        diff = (mean_c-mean_t) / np.sqrt((sd_c**2+sd_t**2)/2)
    return diff
    
def convert_to_formatting(entry_types):
    for entry_type in entry_types:
        if entry_type == 'string':
            yield 's'
        elif entry_type == 'float':
            yield '.3f'
        elif entry_type == 'integer':
            yield '.0f'
    
def add_row(entries, entry_types, col_spans, width):
    vis_cols = len(col_spans)
    invis_cols = sum(col_spans)
    char_per_col = width // invis_cols
    first_col_padding = width % invis_cols
    char_spans = [char_per_col * col_span for col_span in col_spans]
    char_spans[0] += first_col_padding
    formatting = convert_to_formatting(entry_types)
    line = ['%'+str(s)+f for (s,f) in zip(char_spans,formatting)]
    return (''.join(line) % tuple(entries)) + '\n'

def add_line(width):
    return '-'*width + '\n'

class data(Dict):
    def __init__(self, outcome, treatment, covariates):
        Y, D, X = preprocess(outcome,treatment,covariates)
        self._dict = dict()
        self._dict['Y'] = Y
        self._dict['D'] = D
        self._dict['X'] = X
        self._dict['N'], self._dict['K'] = X.shape
        self._dict['controls'] = (D==0)
        self._dict['treated'] = (D==1)
        self._dict['Y_c'] = Y[self._dict['controls']]
        self._dict['Y_t'] = Y[self._dict['treated']]
        self._dict['X_c'] = X[self._dict['controls']]
        self._dict['X_t'] = X[self._dict['treated']]
        self._dict['N_t'] = D.sum()
        self._dict['N_c'] = self._dict['N'] - self._dict['N_t']
        if self._dict['K']+1 > self._dict['N_c']:
            raise ValueError('Too few control units: N_c < K+1')
        if self._dict['K']+1 > self._dict['N_t']:
            raise ValueError('Too few treated units: N_t < K+1')
            
class Summary(Dict):
    def __init__(self, data, varnames, direction):
        self._dict = dict()
        self._dict['N'], self._dict['K'] = data['N'], data['K']
        self._dict['N_c'], self._dict['N_t'] = data['N_c'], data['N_t']
        self._dict['Y_c_mean'] = data['Y_c'].mean()
        self._dict['Y_t_mean'] = data['Y_t'].mean()
        self._dict['Y_c_sd'] = np.sqrt(data['Y_c'].var(ddof=1))
        self._dict['Y_t_sd'] = np.sqrt(data['Y_t'].var(ddof=1))
        self._dict['rdiff'] = self['Y_t_mean'] - self['Y_c_mean']
        self._dict['X_c_mean'] = data['X_c'].mean(0)
        self._dict['X_t_mean'] = data['X_t'].mean(0)
        self._dict['X_c_sd'] = np.sqrt(data['X_c'].var(0, ddof=1))
        self._dict['X_t_sd'] = np.sqrt(data['X_t'].var(0, ddof=1))
        self._dict['ndiff'] = calc_ndiff(self['X_c_mean'],
                                         self['X_t_mean'],
                                         self['X_c_sd'],
                                         self['X_t_sd'],
                                        direction)
        self._dict['varnames'] = varnames
        
    def __str__(self):
        table_width = 80
        N_c, N_t, K = self['N_c'], self['N_t'], self['K']
        Y_c_mean, Y_t_mean = self['Y_c_mean'], self['Y_t_mean']
        Y_c_sd, Y_t_sd = self['Y_c_sd'], self['Y_t_sd']
        X_c_mean, X_t_mean = self['X_c_mean'], self['X_t_mean']
        X_c_sd, X_t_sd = self['X_c_sd'], self['X_t_sd']
        rdiff, ndiff = self['rdiff'], self['ndiff']
        varnames = self['varnames']
        output = '\n'
        output += 'Summary Statistics\n\n'
        entries1 = ['', 'Controls (N_c='+str(N_c)+')','Treated (N_t='+str(N_t)+')', '']
        entry_types1 = ['string']*4
        col_spans1 = [1, 2, 2, 1]
        output += add_row(entries1, entry_types1,col_spans1, table_width)
        entries2 = ['Variable', 'Mean', 'S.d.','Mean', 'S.d.', 'Raw-diff']
        entry_types2 = ['string']*6
        col_spans2 = [1]*6
        output += add_row(entries2, entry_types2,col_spans2, table_width)
        output += add_line(table_width)
        entries3 = ['Y', Y_c_mean, Y_c_sd, Y_t_mean, Y_t_sd, rdiff]
        entry_types3 = ['string'] + ['float']*5
        col_spans3 = [1]*6
        output += add_row(entries3, entry_types3,col_spans3, table_width)
        output += '\n'
        output += add_row(entries1, entry_types1,col_spans1, table_width)
        entries4 = ['Variable', 'Mean', 'S.d.','Mean', 'S.d.', 'Nor-diff']
        output += add_row(entries4, entry_types2,col_spans2, table_width)
        output += add_line(table_width)
        entry_types5 = ['string'] + ['float']*5
        col_spans5 = [1]*6
        for entries5 in zip(varnames, X_c_mean, X_c_sd,X_t_mean, X_t_sd, ndiff):
            output += add_row(entries5, entry_types5,col_spans5, table_width)
        return output

def matching(df,predictors,treatment,k,replace):
    random_state = 100
    log_model_ps = LogisticRegression(random_state = random_state, solver = "saga", max_iter = 100, penalty = "l2")
    fitted_model_ps = log_model_ps.fit(df[predictors],df[treatment])
    propensity = fitted_model_ps.predict_proba(df[predictors])[:,1]
    df_propensity =  pd.DataFrame({'groups':df[treatment], 'propensity':propensity}) 
    N = len(df_propensity)
    N1 = df_propensity[df_propensity["groups"] == 1].index; N2 = df_propensity[df_propensity["groups"] == 0].index
    g1, g2 = df_propensity.iloc[N1,1], df_propensity.iloc[N2,1]
    np.random.seed(random_state)
    morder = np.random.permutation(N1)
    matches = {}
    # k: an integer default is 1.This specifies the k in k nearest neighbors
    # replace: logical for whether individuals from the larger group should be allowed to match multiple individuals in the smaller group.
    for m in morder:
        dist = abs(g1[m] - g2)
        caliper = dist.sort_values().iloc[k-1]
        keep = np.array(dist[dist<=caliper].index)
        if len(keep):
            matches[m] = keep
        else:
            matches[m] = [dist.argmin()]
        if not replace:
            g2 = g2.drop(matches[m])
    tr = matches.keys()
    ctrl = [m for matchset in matches.values() for m in matchset]
    df_matched = pd.concat([df.loc[tr,:], df.loc[ctrl,:]])
    # remove duplicate rows
    df_matched = df_matched.drop_duplicates()
        
    return [df_propensity,matches,df_matched];

def matching_rf(df,predictors,treatment,k,replace):
    random_state = 100
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    max_depth = [int(x) for x in np.linspace(2, 40, num = 10)]
    max_features = ['auto', 'sqrt', 'log2']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4, 10]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=100, n_jobs = -1)
    rf_random.fit(df[predictors],df[treatment])
    params = rf_random.best_params_
    rf_randomBest = RandomForestClassifier(**params)
    rf_randomBest.fit(df[predictors],df[treatment])
    propensity = rf_randomBest.predict_proba(df[predictors])[:,1]
    #import scikitplot as skplt
    #probas_list = [y_pred_prob]
    #skplt.metrics.plot_calibration_curve(df[col_treatment],probas_list)
    #plt.rcParams['figure.figsize'] = 8, 6
    #plt.rcParams['font.size'] = 12
    #sns.distplot(neg_pscore, label='control')
    #sns.distplot(pos_pscore, label='treatment')
    #plt.xlim(0, 1)
    #plt.title('Propensity Score Distribution of Control vs Treatment')
    #plt.ylabel('Density')
    #plt.xlabel('Scores')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
    df_propensity =  pd.DataFrame({'groups':df[treatment], 'propensity':propensity}) 
    N = len(df_propensity)
    N1 = df_propensity[df_propensity["groups"] == 1].index; N2 = df_propensity[df_propensity["groups"] == 0].index
    g1, g2 = df_propensity.iloc[N1,1], df_propensity.iloc[N2,1]
    np.random.seed(random_state)
    morder = np.random.permutation(N1)
    matches = {}
    # k: an integer default is 1.This specifies the k in k nearest neighbors
    # replace: logical for whether individuals from the larger group should be allowed to match multiple individuals in the smaller group.
    for m in morder:
        dist = abs(g1[m] - g2)
        caliper = dist.sort_values().iloc[k-1]
        keep = np.array(dist[dist<=caliper].index)
        if len(keep):
            matches[m] = keep
        else:
            matches[m] = [dist.argmin()]
        if not replace:
            g2 = g2.drop(matches[m])
    tr = matches.keys()
    ctrl = [m for matchset in matches.values() for m in matchset]
    df_matched = pd.concat([df.loc[tr,:], df.loc[ctrl,:]])
    # remove duplicate rows
    df_matched = df_matched.drop_duplicates()
        
    return [df_propensity,matches,df_matched];
