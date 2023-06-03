# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:30:10 2022

@author: Rayanet
"""

########### EDA ###########


#pandas profiling
"""
from pandas_profiling import ProfileReport
profile = ProfileReport(*data* , title="**", explorative=True)
profile.to_file("*name*.html")
"""

#sweetviz
"""
import sweetviz as sv
my_report = sv.analyze(*data file name*, traget_feat= '*target feacher name')
my_report.to_file("*name*.html")

Compare 2 dataset wits sweetviz
"""



#pandas GUI
from pandas_profiling import ProfileReport
import pandas as pd
data = pd.read_excel("Master_data.xlsx")
profile = ProfileReport(data , title="corona", explorative=True)
profile.to_file("corona.html")