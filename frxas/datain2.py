import numpy as np
import pandas as pd
import glob 
import os
from matplotlib import pyplot

#Making array of strings to help pandas find all data files
# for a single measured amplitude and location (point)
def getdata(P, A, base, initfile, finfile):
    
    filestr=str(P + ' ' + str('%.3f'%float(A)) + ' '+ "*0*.txt")
    
	all_files=glob.glob(os.path.join(base, filestr))
	return allfiles
	