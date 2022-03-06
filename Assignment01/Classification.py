# %% Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join



# %% Colecting LS Data

LSpath = "./data/Group16/Classification/LS_Group16/"
classes = ["Class1.txt", "Class2.txt", "Class3.txt"]

dfs = []

for i, c in enumerate(classes):
	df = pd.read_csv(join(LSpath, c), names = ["x", "y"], sep = " ", dtype = {"x" : "float", "y" : "float"}, engine = "python")
	df["class"] = i 
	dfs.append(df)

ls_data = pd.concat(dfs, axis=0).reset_index(drop=True) 




# %% Collecting NLS Data

nls_data = pd.read_csv("./data/Group16/Classification/NLS_Group16.txt", names = ["x", "y"], sep = " ", 
					dtype = {"x" : "float", "y" : "float"}, engine = "python", skiprows = 1, index_col = False)

nls_data["class"] = [0]*500 + [1]*500 + [2]*500




# %%

