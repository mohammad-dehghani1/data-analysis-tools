# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:30:10 2022

@author: Rayanet
"""

<<<<<<< HEAD



=======
>>>>>>> 9af56298869de8a3b999af2e8edcfcf2cc7988e4
import pandas as pd
import seaborn as sns
import sweetviz as sv

# Compute the correlation matrix
df = pd.read_csv("dataset_for_stair.csv")
df = df.iloc[:,:-1]

def analyze(report_kind):
    '''
    kinds: corr_matrix, whole_analyze

    '''
    report_kind = report_kind
    if report_kind == 'corr_matrix':
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    if report_kind == 'whole_analyze':
        my_report = sv.analyze(df, target_feat= 'amnt')
        my_report.show_html()
<<<<<<< HEAD

=======
>>>>>>> 9af56298869de8a3b999af2e8edcfcf2cc7988e4
