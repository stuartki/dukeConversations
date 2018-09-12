import csv
import operator
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from itertools import islice
import math
import networkx as nx

dataSpring2018 = pd.ExcelFile('spring2018.xlsx').parse(0)
dataFall2017 = pd.ExcelFile('fall2017.xlsx').parse(0)

##rating INDICTS
majIndict = 1
attIndict = 1
count5Indict = 1


#returns dukeConvoosParameters in dict, dataSpring2018frame
def dictifier2018():
	dateDict = {}
	with open('dukeConvoosParameters.csv', 'rb') as csvfile:
		dukeConvoosParameters = csv.reader(csvfile)
		
		for row in islice(dukeConvoosParameters,85,122):
			dateDict[row[0]] = row[1:8]
	paraSpring2018 = pd.read_csv('dukeConvoosParameters.csv', skiprows = 84, skipfooter = 31, engine = 'python')
	paraSpring2018 = paraSpring2018.iloc[:,:9]
	paraSpring2018['Date'] = pd.to_datetime(paraSpring2018['Date'], format = '%m/%d/%y')
	return dateDict, paraSpring2018
	
def dictifier2017():
	dateDict = {}
	with open('dukeConvoosParameters.csv', 'rb') as csvfile:
		dukeConvoosParameters = csv.reader(csvfile)
		
		for row in islice(dukeConvoosParameters,125,152):
			dateDict[row[0]] = row[1:8]
	paraSpring2018 = pd.read_csv('dukeConvoosParameters.csv', skiprows = 124)
	paraSpring2018 = paraSpring2018.iloc[:,:9]
	return dateDict, paraSpring2018

#cleans duplicates in certain col_name
def cleaner(data, col_name):
	re = data.drop_duplicates(subset = col_name, keep = 'first')
	re.to_csv('fall2017test.csv', index = False)

#distribution of all majors
def majorGrapher():
	ti = dataSpring2018['major'] 
	ticc = Counter(ti)
	
	df = pd.dataFrame.from_dict(ticc, orient='index')
	
	#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	
	df.plot(kind='bar', fontsize = 10)

	#annoying label changer
	#condition for nan
	locs, labs = plt.xticks()
	ti = dataSpring2018['major']
	newLabs = ['nan']
	newLabs.extend([labeler(ti)[x.get_text()] for x in labs if x.get_text() !='nan'])
	plt.xticks(locs, newLabs)


	plt.tight_layout()	
	plt.style.use('ggplot')
	plt.show()
	
#shorter label markers
def labeler(series):

	dict = {}
	for x in series.unique():
		if x == "Biomedical Engineering":
			dict[x] = "BME"
			continue
		dict[x] = str(x)[:2]
	return dict
	
#distribution of times by month
def timeGrapher():
	ti = dataSpring2018['timestamp']
	tiMonth = [datetime.strptime(n, '%m/%d/%y %H:%M').day for n in ti]
	ticc = Counter(tiMonth)
	
	df = pd.dataFrame.from_dict(ticc, orient='index')
	
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
	
	df.plot(kind='bar', fontsize = 5)
	plt.show()

#distribution of majors by certain dinners
def facultyMajorDist(num = 0, facList = []):

	count = 1
	dict, para = dictifier2018()

	
	if num == 0 and len(facList) == 0:
		
		num = para.shape[0]
		list = para['FacultyKey']
	elif len(facList) == 0:
		list = para['FacultyKey'].head(num)
	else:
		list = facList
		num = len(facList)
	
	

	for x in list:
		plt.style.use('ggplot')
		plt.subplot(num, 1, count)

		facDist = dataSpring2018[dataSpring2018['prof'] == x]['major'].value_counts()
		facDist.plot(kind = 'bar')
		plt.title(x + ": " + dict[x][0])
		plt.legend()
		
		
		#change labels
		locs, labs = plt.xticks()
		ti = dataSpring2018['major']
		newLabs = [labeler(ti)[x.get_text()] for x in labs]
		plt.xticks(locs, newLabs)

		count += 1
	plt.tight_layout()
	plt.show()
	return facDist

#return professors of "x" department
def departmentFinder(dep):
	para = dictifier2018()[1]
	para['strsplit'] = para.Department.str.split("|")
	para['trueFalse'] = para['strsplit'].apply(lambda x: 'true' if dep in x else 'false') == 'true'
	profsDep = para[para['trueFalse']]
	return profsDep

def facRelaterRating(G, faculty = "all"):

	facRatingDict = {}
	facRating = 0
	para = dictifier2018()[1]
	if faculty == "all":
		faculty = para['FacultyKey'].unique()


	paraSet = para.set_index('FacultyKey')
	fac = paraSet.loc[faculty]
	
	d = pd.merge(left = fac, right = dataSpring2018, left_on = 'FacultyKey', right_on = 'prof').set_index('prof')


	for f in faculty:

		tempD = d.loc[f]
		for ind, x in tempD.iterrows():
			dep = x['Department']
			maj = x['major']
			if isinstance(maj, float) or isinstance(dep, float):
				continue
				
			##### METHOD
			if count5Indict == 1:
				if G[dep][maj]['count'] < 5:
					continue
			##### METHOD
			facRating += G[dep][maj]['weight']
		facRatingDict[f] = round(float(facRating), 2)
	return facRatingDict
	
def relationsMajorGrapher():
	G = nx.Graph()

	para = dictifier2018()[1]
	para2 = dictifier2017()[1]

	d = pd.merge(right = para, left = dataSpring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = dataFall2017, right_on = 'FacultyKey', left_on = 'prof')
	da = pd.concat([d, d2], sort = True)

	facD =  da['major'].value_counts()

	count = 0
	for index, x in da.iterrows():
		maj = str(x['major'])
		dep = str(x['Department'])

		if maj == 'nan' or dep == 'nan':
			continue
		
		##### METHOD	
		if majIndict == 1:	
 			normalize = float(facD.loc[maj])
		else:
			normalize = 1.
		#####METHOD


		if G.has_edge(dep, maj):
			G[dep][maj]['weight'] += 1./normalize
			G[dep][maj]['count'] +=1
			if G[dep][maj]['count'] > 5:
				count +=1
		else:
			G.add_edge(dep, maj, weight = 1./normalize)
			G.add_edge(dep, maj, count = 1)
	return G

def dictSorterItems(x, rev = ""):
	if rev == "reverse":
		r = True
	else:
		r = False
	sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse = r)
	return sorted_x

def attIndictRater(G):
	refSeries = dataSpring2018['prof'].value_counts()
	total = 0.
	for x in refSeries:
		total += float(x)
	print total
	sorted_dick = dictSorterItems(facRelaterRating(G), rev = "reverse")
	for x in range(len(sorted_dick)):
		prof = sorted_dick[x][0]
		sorted_dick[x] = (prof, round(total*sorted_dick[x][1]/float(refSeries.loc[prof]), 2), refSeries.loc[prof])
		

	s = sorted(sorted_dick, key=operator.itemgetter(1), reverse = True)

	return s
	
def firstAcceptances(data, plot = False):
	s = set()
	refSeries = data['prof'].value_counts()

	delIndex = []
	data = data.reset_index()
	data = data.sort_values(by = ['timestamp'])
	for index, row in data.iterrows():
		if row['unique id'] in s:
			delIndex.append(index)
		else:
			s.add(row['unique id'])

	d = data.drop(delIndex)['prof'].value_counts()


	ha = {}
	for index, value in d.items():
		ha[index] = float(value)/float(refSeries[index])
	fA = dictSorterItems(ha, rev = "reverse")
	
	if plot:
		d.plot.bar()
		plt.tight_layout()
		plt.style.use('ggplot')
		plt.show()
	
	return ha

def dinner_info(fac, plot_major = False):

	##DATA
	
	dinner = dataSpring2018[dataSpring2018['prof'] == fac]
	dep = para[para['FacultyKey'] == fac]['Department'].values[0]
	G = relationsMajorGrapher()
	s = [(n[0], round(n[1], 2)) for n in sorted([(n, G[dep][n]['weight']) for n in G.neighbors(dep)], key = operator.itemgetter(1), reverse = True)[:5]]
	raterDict = {}
	r = attIndictRater(G)
	fADict = firstAcceptances(dataSpring2018) 
	max = r[0][1]

	for x in attIndictRater(G):
		print x
		raterDict[x[0]] = round(x[1]/max, 2)


	##DATA
	
	print para[para['FacultyKey'] == fac]['FacultyName'].values[0]
	print "Department: " + para[para['FacultyKey'] == fac]['Department'].values[0]
	print "Number of Students: " + str(len(dinner))
	print "Date: " + str(para[para['FacultyKey'] == fac]['Date'].values[0])
	print "Related Majors: " + str(s)
	print "Major Diversity Rating Score: " + str(raterDict[fac])
	print "First Acceptance Score: " + str(round(fADict[fac], 2))
 	
 		
	if plot_major:
		plt.style.use('ggplot')
		dinner['major'].value_counts().plot.bar()
# 		locs, labs = plt.xticks()
# 		ti = dataSpring2018['major']
# 		newLabs = [labeler(ti)[x.get_text()] for x in labs]
# 		plt.xticks(locs, newLabs)
		plt.tight_layout()
		plt.show()

##WORKSPACE


indict = 4
if indict == 1:
	# print dataSpring2018.info()
	
	d, para = dictifier2018()
	# print para.info()
	
	department = "Computer Science"
	
	facultyMajorDist(num = 3)
	
	#dinner_info('hartemink', plot_major = True)
	
	
	
	
	
	
 	#facultyMajorDist(facList = departmentFinder(department)['FacultyKey'].values)
	
if indict == 2:
	
	G = relationsMajorGrapher()
	dep = 'English'
	s =  sorted([(n, G[dep][n]['weight']) for n in G.neighbors(dep)], key = operator.itemgetter(1), reverse = True)
	print s
	print attIndictRater(G)
	
	
if indict == 3:
	d, para = dictifier2018()
 	para['Date'] = pd.to_datetime(para['Date'], format = '%m/%d/%y')
 	para = para.sort_values(by = ['Date'])

	print para[['FacultyKey', 'Date']].head(10)

	s = set()
	refSeries = dataSpring2018['prof'].value_counts()
	delIndex = []
	dataSpring2018 = dataSpring2018.reset_index()
	dataSpring2018 = dataSpring2018.sort_values(by = ['timestamp'])

	for index, row in dataSpring2018.iterrows():
		if row['unique id'] in s:
			delIndex.append(index)
		else:
			s.add(row['unique id'])

	d = dataSpring2018.drop(delIndex)['prof'].value_counts()


	ha = {}
	for index, value in d.items():
		ha[index] = round(float(value)/float(refSeries[index]), 2)
	
	y = []
	for index, row in para.iterrows():
		y.append(ha[row['FacultyKey']])
		print row['FacultyKey'] + ": " +  str(ha[row['FacultyKey']])
	plt.plot(y)

	
# 	d.plot.bar()
	plt.tight_layout()
	plt.style.use('ggplot')
	plt.show()
	
if indict == 4:
	d, para = dictifier2018()
	dinner_info('adair')

##WORKSPACE










