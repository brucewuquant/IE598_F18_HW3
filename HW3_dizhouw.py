# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 23:08:40 2018

@author: wdz
"""
# You do not have to do the other datasets (abalone, winequality or glass)!
import pylab
import scipy.stats as stats
import urllib.request
import sys
import numpy as np

#read from uci data
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib.request.urlopen(target_url)

#arrange data into list for labels and list of lists for attributes
xList =  []
labels = []
for line in data:
    #split on comma
    row = bytes.decode(line).strip().split(",")
    xList.append(row)

#sys.stdout.write('number of rows of data = ' + str(len(xList))+ '\n')
#sys.stdout.write('number of cols of data =' + str(len(xList[1])))

nrow = len(xList)
ncol = len(xList[1])

type = [0]*3
colCounts = []

'''

for col in range(ncol):
    for row in xList:
        try: 
            a = float(row[col])
            if isinstance(a, float):
                type[0]+=1
        except ValueError:
            if len(row[col])> 0:
                type[1]+=1
            else:
                type[2] += 1
    colCounts.append(type)
    type=[0]*3

sys.stdout.write('Col#' +'\t' + 'Number' + '\t' +
                 'Strings' + '\t' + 'Other\n')

iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol +=1

'''
#generate summary statistics for column 3(e.g)

col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))

'''
colArray = np.array(colData)
colMean = np.mean(colArray)
colStd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean)+
                 '\t\t'+'Standard Deviation = ' +
                 '\t' + str(colStd) + "\n")


#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, ( i *100)/ntiles ))

sys.stdout.write("Boundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write("\n")

# run again w/ 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write("\n")

col  = 60
colData = []
for row in xList:
    colData.append(row[col])

unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

#count up the number of elements having each value

catDict = dict(zip(list(unique), range(len(unique))))

catCount = [0]*2

for elt in colData:
    catCount[catDict[elt]] += 1

sys.stdout.write('\nCounts for each value of categorical label')

print(list(unique)) 
print(catCount)
'''
stats.probplot(colData, dist = "norm", plot = pylab)
pylab.show()

#%%
import pandas as pd
import matplotlib.pyplot as plot
target_url = ( "https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    
rocksVMines = pd.read_csv(target_url, header = None, prefix = "V")

#print head and tail of dataframe
print(rocksVMines.head())
print(rocksVMines.tail())

summary = rocksVMines.describe()
print(summary)

# parallel coordinates graph for real attribute visualization -linePlot

#%%
import pandas as pd
import matplotlib.pyplot as plot
target_url = ( "https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    
rocksVMines = pd.read_csv(target_url, header = None, prefix = "V")
# iat: Similar to iloc, in that both provide integer-based lookups. Use iat if you only need to get or set a single value in a DataFrame or Series.

for i in range(208):
    # asign color based on "M" or "R" label
    if rocksVMines.iat[i, 60] == 'M':
        pcolor = "red"
    else: 
        pcolor = "blue"
    
    #plot rows of data as of they were series data
    dataRow = rocksVMines.iloc[i, 0:60]
    dataRow.plot(color = pcolor)

plot.xlabel('Attribute Index')
plot.ylabel('Atribute Values')
plot.show()

#%%

import pandas as pd
import matplotlib.pyplot as plot
target_url = ( "https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    
rocksVMines = pd.read_csv(target_url, header = None, prefix = "V")
# iat: Similar to iloc, in that both provide integer-based lookups. Use iat if you only need to get or set a single value in a DataFrame or Series.

dataRow2 = rocksVMines.iloc[1, 0:60]
dataRow3 = rocksVMines.iloc[2, 0:60]

plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()
dataRow21 = rocksVMines.iloc[20,0:60]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()

#%%
import pandas as pd
import matplotlib.pyplot as plot
target_url = ( "https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    
rocksVMines = pd.read_csv(target_url, header = None, prefix = "V")
# iat: Similar to iloc, in that both provide integer-based lookups. Use iat if you only need to get or set a single value in a DataFrame or Series.

#change the targets to numeric values
target = []
for i in range(208):
    #assign 0 or 1 target value base on "M" or "R"
    if rocksVMines.iat[i, 60] == "M":
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))

#plot 35th attribute
dataRow = rocksVMines.iloc[0:208, 35]
plot.scatter(dataRow, target)

plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#%%
import sys
import pandas as pd
import matplotlib.pyplot as plot
from math import sqrt
target_url = ( "https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    
rocksVMines = pd.read_csv(target_url, header = None, prefix = "V")
# iat: Similar to iloc, in that both provide integer-based lookups. Use iat if you only need to get or set a single value in a DataFrame or Series.

#calculate corr 
dataRow2 = rocksVMines.iloc[1, 0:60]
dataRow3 = rocksVMines.iloc[2, 0:60]
dataRow21 = rocksVMines.iloc[20, 0:60]

mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i] / numElt
    mean21 += dataRow21[i]/ numElt

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/ numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/ numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/ numElt
    
corr23=  0.0 ; corr221 = 0.0
for i in range(numElt):
    corr23+=(dataRow2[i] - mean2)*(dataRow3[i] - mean3)/(sqrt(var2*var3) * numElt)
    corr221 +=(dataRow2[i] - mean2)*(dataRow21[i] - mean21)/ (sqrt(var2*var21)*numElt)

sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")

#%%
# Visualizing Attribute and Label Correlations Using a Heat Map
import pandas as pd
import matplotlib.pyplot as plot
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url,header=None, prefix="V")

#calculate corr betn real-val attributes
corMat = pd.DataFrame(rocksVMines.corr())
plot.pcolor(corMat)
plot.show()


print("My name is WU DIZHOU")
print("My NetID is: dizhouw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")





































































