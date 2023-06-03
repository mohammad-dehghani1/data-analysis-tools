import pandas as pd

df = pd.read_csv('directory')

####################
# exploring data
####################
df.head()

df.tail()

df.shape()

df.describe()

df.info()

df.columns()


df[ ['column1'], ['column2'] ]

df [ 'culumn1'] # does not return data frame

df.sample( 5, random_state =0)

######################
#filtering data
######################

# filter on integer
salary_gt_x = df['column1'] > 200000 #bolean arrey
df[salary_gt_x]
# or
df[ df['column1'] > 200000]

#filter on string
df[ df['text_column'] == 'text' ]

#find texts containing a string
title = df['text_column'].str.contains('text term')
df [title]

######################
# visualization
######################

#histogram
df['Salary'].plot.hist(
    bins=100,
    title='Histogram of Salaries', 
    xlim=[0,200000], 
    figsize=(10,6)
);

# scatter plot
df.plot.scatter(
    x='Seniority',
    y='Salary',
    title='Scatter plot Salary v. Seniority',
    figsize=(10,6)
);

# bar plot
df.plot.bar(
    x='Seniority',
    y='Seniority_count',
    title='Count of Persons by Seniority',
    figsize=(10,6),
);


# computing summery
import pandas as pd
df = pd.read_csv('data')




df = df[ ['plateclass', 'amnt']]

df.groupby('plateclass').mean()
df.groupby('plateclass').agg(['mean','median','count'])

#######
#data cleaning

#sorting
df = df.sort_values(by=['Seniority','Salary'],ascending=[False,True])

#missing value
df['title'] = df['title'].fillna('text') # this code, replaces the 'text' into NAL value in 'title' column

df.dropna() # this code removes the rows with NAL value
df.dropna(subset=['title']) # this only drops a row if the 'title'column has a missing value
df.dropna(subset=['Title','Salary']) # This drops a row if the 'Title' or 'Salary' columns (at least one) have a missing value.
df.dropna(subset=['Title','Salary'], how='all') # This drops a row only if both (all of) 'Title' and 'Salary' are missing a value

df.loc[10, 'title'] = 'text' # this insert 'text' to row 10 column 'title'


df['title'].isna()
df['title'].notna()

df['title'] = df['title'].replace('current text' : 'new text')

#dummy table 
pd.get_dummies(df,columns=['plateprvnc'],prefix='province') # prefix adding a text before dummy columns

#creating data frame
herd_dict = {
    "name" : ["Bluebell", "Daisy", "Nellie"],  #one column in data frame
    "breed" : ["Holstein", "Jersey", "Holstein"],
    "weight" : [1305, 807, 1296],
    "GPD"    : [8.9, 5.8, 9.1]
}

###########
# indexing
df = df.set_index('clmn_name')

df.loc['rowname, clmnname']

df.iloc[2,1]


