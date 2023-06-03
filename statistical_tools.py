# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:38:54 2023

@author: Rayanet
"""
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


table_name = pd.read_csv('directory.csv')




################## Estamates of location ##################

# Mean
table_name['column_name'].mean()

# Median
table_name['column_name'].Median()

# Trim mean
from scipy import state
state.trim_mean(table_name['column_name'], 0.1)

# Weight median
np.average(table_name['column_name'], weights= table_name['column_name'])
# or
wquantiles.median(table_name['column_name'], weights= table_name['column_name'])



################## Estamets of variability ##################

# Standard deviaion
table_name['column_name'].std()

# IQR (InterQuantile range)
table_name['column_name'].quantile(0.75) - table_name['column_name'].quantile(0.25)

# MAD (Mean absolute deviation)
robust.scale.mad(table_name['column_name'])



################## Exploring the data distribution ##################

# boxplot
ax = (table_name['column_name']).plot.box()
ax.set_ylabel('y_label_name')

# Frequency table
fx = pd.cut(table_name['column_name'], number_of_bins)
fx.value_counts()

# Histogram plot
ax = table_name['column_name'].plot.hist()

# Density plot
table_name['column_name'].plot.density()



################## Explorinf binary and categorical data ##################

# Bar chart
ax = table_name['column_name'].plot.bar()



################## Correlation ##################

# Correlation matrix
table_name[['clmn1'],['clmn1'],['...']].corr()

# Scatter plot
table_name.plot.scatter(x= '', y= '')



################## Exploring two or more variable ##################

# Hexagonal plot
ax = table_name.plot.hexbin(x='',y='')

# Contour plot
ax = sns.kdeplot(table_name.column_name, table_name.column_name, ax = ax) #not working

# Heatmap


# Two categorical variables

#Pivot Table

# Violin Plot
ax = sns.violinplot(tb_name.cb_name ,tb_name.cb_name)


######################################################
################## Sampling distribution ##################
######################################################



# compare Sampling data with population
sample_data = pd.DataFrame ({
    'clmn_name' : table_name.sample(num_of_sample)
    'clmn2_name' : 'Label_1',
    })
sample_mean_5 = pd.DataFrame({
    'clmn_name' : [table_name.sample(5).mean() for _ in range(1000)],
    'clmn2_name' : 'Label_2',
    }) 

results = pd.concat([sample_data, sample_mean_5])
g = sns.FaceGrid(results, col='clmn2_name')
g.map(plt.hist, 'clmn_name', range= [ , ], bins= '')

# bootstrap
R_resutls =[]
for R_repeat in range(1000):
    sample = resample(table_name)
    R_results.append(sample.median())
R_results = pd.Series(R_sesults)
bias = (R_sesults.mean - table_name.median())
std_error = R_results.std()

# normalizing data
data = table_name.column_name
normal_dataset =[]
data_mean = data.mean()
data_Std = data.std()
for x in data:
    normal_data = (x - data_mean) / data_Std
    normal_dataset.append(normal_data)
    
# QQ-plot
fig, ax = plt.subplot(figsize = (4,4))
sp.stats.probplot(data_table, plot=ax)

# Binomial distribution
sp.stats.binom.pmf(x , n= , p= )
# or
sp.stats.binom.cdf(x , n= , p= )

# Poisson distribution
sp.stats.poisson.rvs('lambda', size = ' ')

# Exponential distribution
sp.stats.expon.rvs('lambda', size = )




################## Resampling ##################

# Boxplot for A/B testing
# e.g: 2 webpage
ax = table_name.boxplot(by= 'page', column='time')

# Mean for A/B testing
mean_a = table_name[tb_name.clmn_name == 'pageA'].Time.mean()
mean_b = table_name[tb_name.clmn_name == 'pageB'].Time.Mean()
diff = mean_a - mean_b

# Promuatation test & p-value
mean_a = table_name[tb_name.clmn_name == 'pageA'].Time.mean()
mean_b = table_name[tb_name.clmn_name == 'pageB'].Time.Mean()
observed_diff = mean_a - mean_b
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()
perm_diffs = [perm_fun(tb_name.clmn_name, nA, nB) for x in range(1000)]
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(perm_diffs, bins=11, rwidth= 0.9)
ax.axvline(x = mean_b - mean_a, color='black', lw=2)
ax.text(50, 190, 'Observed\ndifference', bbox={'facecolor':'white'})
ax.set_xlabel('Session time differences (in seconds)')
ax.set_ylabel('Frequency')
p_value = np.mean([diff > observed_dif for diff in perm_diffs])

# p-value
A = 'sucsess of population a' #e.g: convertion of price A
n_A = 'poppulation A' #e.g: all visitors of price A
B = 'sucsess of population B' #e.g: convertion of price B
n_B = 'poppulation B'#e.g: all visitors of price B
survivors = np.array([[A, n_A - A], [B, n_B - B]])
chi2, p_value, df, _ = sp.stats.chi2_contigency(survivors)
    
# t-test
res = stats.ttest_ind(session_times[session_times.Page == 'Page A'].Time,
session_times[session_times.Page == 'Page B'].Time,
equal_var=False)


################## Anova ##################

# Premutaion test
observed_variance = four_sessions.groupby('Page').mean().var()[0]
print('Observed means:', four_sessions.groupby('Page').mean().values.ravel())
print('Variance:', observed_variance)
def perm_test(df):
    df = df.copy()
    df['Time'] = np.random.permutation(df['Time'].values)
    return df.groupby('Page').mean().var()[0]
perm_variance = [perm_test(four_sessions) for _ in range(3000)]
print('Pr(Prob)', np.mean([var > observed_variance for var in perm_variance]))

# sample size
effect_size = sm.stats.proportion_effectsize(0.0121, 0.011)
analysis = sm.stats.TTestIndPower()
result = analysis.solve_power(effect_size=effect_size,
                              alpha=0.05, power=0.8, alternative='larger')
print('Sample Size: %.3f' % result)



######################################################
################## Regression and prediction ##################
######################################################

################## linear regression ##################

from sklearn.linear_model import LinearRegression
predictors = ['X1', 'X2', '...']
outcome = 'Y'
model = LinearRegression()
model.fit(data_set[predictors], data_set[outcome])
print(f'Intercept: {model.intercept_:.3f}')
for name, coef in zip(predictors, model.coef_):
    print (f' {name}: {coef}')
# fitted value and residuals
fitted = model.predict(lung[predictors])
residuals = lung[outcome] - fitted

# RMSE (root mean squared error)
fitted = model.predict(data_set[predictors])
RMSE = np.sqrt(mean_squared_error(data_set[outcome], fitted))
r2 = r2_score(data_set[outcome], fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')

# Detailed analysis of the regression model
model= sm.OLS(data_set[outcome], house[predictors].assign(conts=1))
results = model.fit()
results.summery()

# Convert categorical variables to dummies variables
pd.get_dummies(data_set['column_name'], drop_first= True)

# Interactions and Main Effects
model = smf.ols(formula= 'target ~ X1* X2 + X3 + ' +
'Bathrooms + Bedrooms + BldgGrade + PropertyType', data=data_set)
results = model.fit()
results.summary()


################## Polynomial and Spline Regression ##################

# Polynomial regression
model_poly = smf.ols(formula='AdjSalePrice ~ SqFtTotLiving + ' +
'+ I(SqFtTotLiving**2) + ' +
'SqFtLot + Bathrooms + Bedrooms + BldgGrade', data=house_98105)
result_poly = model_poly.fit()
result_poly.summary()

# spiline regression
formula = 'AdjSalePrice ~ bs(SqFtTotLiving, df=6, degree=3) + ' +
'SqFtLot + Bathrooms + Bedrooms + BldgGrade'
model_spline = smf.ols(formula=formula, data=house_98105)
result_spline = model_spline.fit()

# GAM (Generalized additive models)
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
X = house_98105[predictors].values
y = house_98105[outcome]
gam = LinearGAM(s(0, n_splines=12) + l(1) + l(2) + l(3) + l(4))
gam.gridsearch(X, y)
