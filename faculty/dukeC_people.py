import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import operator

dataSpring2018 = pd.ExcelFile('spring2018.xlsx').parse(0)
dataSpring2018['Semester'] = "Spring2018"
dataFall2017 = pd.ExcelFile('fall2017.xlsx').parse(0)
dataFall2017['Semester'] = "Fall2017"
dataFall2018 = pd.ExcelFile('fall2018.xlsx').parse(0)
dataFall2018['Semester'] = "Fall2018"
totalData = pd.concat([dataSpring2018, dataFall2017, dataFall2018], sort = True)

d = pd.Series(totalData['unique id'].value_counts())
print len(d)
da = totalData.merge(d, left_on = 'unique id', right_index = True)
print Counter(da[da['unique id_y'] > 1]['unique id'].unique()['year'])

## 421 participants out of 1095 total have more than 2 applications
def dictSorter(x, rev = ""):
	if rev == "reverse":
		r = True
	else:
		r = False
	sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse = r)
	return sorted_x
	
def firstAcceptances(data, plot = False):
	# a score of 1 would be perfect first attendance AND an attendance of dinner of 1
	# not as "valuable" if applied many times
	s = set()
	refSeries = data['prof'].value_counts()
	studentCount = data['unique id'].value_counts()
	
	delIndex = []
	data = data.sort_values(by = ['timestamp'])
	data = data.reset_index()
	
	#creates new column with normalized by how many attendances
	firstAvalue = pd.Series(np.ones(len(data)), index=data.index)

	for index, row in data.iterrows():

		firstAvalue[index] = 1./float(studentCount[row['unique id']])
	data['firstAvalue'] = firstAvalue
	
	
	
	ha = defaultdict(int)
	
	
	
	for index, row in data.iterrows():
		ha[row['prof']] += row['firstAvalue']/refSeries[row['prof']]
	fA = dictSorter(ha, rev = "reverse")
	if plot:
		pd.Series(ha).plot.bar()
		plt.tight_layout()
		plt.show()
	
	return ha, data







	
