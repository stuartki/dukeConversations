import csv
import operator
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from itertools import islice
import math
import networkx as nx
from algorithms import isSimilar
import math

dataSpring2018 = pd.ExcelFile('spring2018.xlsx').parse(0)
dataFall2017 = pd.ExcelFile('fall2017.xlsx').parse(0)
dataFall2018 = pd.ExcelFile('fall2018.xlsx').parse(0)
totalData = pd.concat([dataSpring2018, dataFall2017, dataFall2018], sort = True)

##rating INDICTS
majIndict = 1
attIndict = 1
count5Indict = 1

path = '/Users/stuartki/Dropbox/dukeConversations/'

#returns dukeConvoosParameters in dict, dataSpring2018frame
def dictifierSpring2018():
	paraSpring2018 = pd.ExcelFile(path + 'Spring2018Parameters.xlsx').parse(0)
	return paraSpring2018
	
def dictifierFall2017():
	paraFall2017 = pd.ExcelFile(path + 'Fall2017Parameters.xlsx').parse(0)
	return paraFall2017

def dictifierFall2018():
	paraFall2018 = pd.ExcelFile(path + 'Fall2018Parameters.xlsx').parse(0)
	return paraFall2018

#cleans duplicates in certain col_name
def cleaner(data, col_name):
	re = data.drop_duplicates(subset = col_name, keep = 'first')
	re.to_csv('fall2017test.csv', index = False)

#shorter label markers
def labeler():
	ti = set(dataFall2018['major'].unique())
	ti.update(dataSpring2018['major'].unique())
	dict = {}
	for x in ti:
		if x == "Biomedical Engineering":
			dict[x] = "BME"
			continue
		dict[x] = str(x)[:2]
	return dict

def newLabels(labs):
	newLabs = [labeler()[x.get_text()] for x in labs if x.get_text() !='nan']
	newLabs.append('nan')
	return newLabs
	
def dictSorter(x, rev = ""):
	if rev == "reverse":
		r = True
	else:
		r = False
	sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse = r)
	return sorted_x
	
	
def getUniqueID(student):
	uID = 0
	for index, row in totalData.iterrows():
		if isSimilar(student, row['name']):
			print row['name']
			inp = raw_input("yes?")
			if inp == "y" or inp == "yes":
				uID = row['unique id']
				break
	if uID == 0:
		return None
	return uID

def getStudent(student):
	uID = getUniqueID(student)
	if uID == None:
		return None
	else:
		l = totalData[totalData['unique id'] == uID][['year', 'why', 'prof', 'timestamp']]
		print l
		return l
	
#distribution of all majors
def majorGrapher(data):
	ti = data['major'] 
	ticc = Counter(ti)
	
	df = pd.DataFrame.from_dict(ticc, orient='index')
	
	#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	
	df.plot(kind='bar', fontsize = 10)

	#annoying label changer
	locs, labs = plt.xticks()
	plt.xticks(locs, newLabels(labs))
	plt.tight_layout()
	plt.show()
	
#return professors of "x" department
def departmentFinder(dep):
	para = dictifierSpring2018()
	para['strsplit'] = para.Department.str.split("|")
	para['trueFalse'] = para['strsplit'].apply(lambda x: 'true' if dep in x else 'false') == 'true'
	profsDep = para[para['trueFalse']]
	return profsDep
	
#distribution of times by dinner
def timeDinner(data, fac = ""):
	ti = data['timestamp']
	tiMonth = [n.to_pydatetime() for n in ti]
	if fac == "":
		for n in data['prof'].unique():
			raw_input(n)
			dinner = data[data['prof'] == n]
			dinner.groupby(dinner["timestamp"].dt.hour)["timestamp"].count().plot(kind="bar")
			plt.title(n + ": First Time = " + str(min(dinner["timestamp"])))
			plt.show()
	else:
		dinner = data[data['prof'] == fac]
		dinner.groupby(dinner["timestamp"].dt.hour)["timestamp"].count().plot(kind="bar")
		plt.title(fac)
		plt.show()

#by hour, distribution of times applied within 
def timeSummary(data, plot = False, withinDay = 1):
	dict = defaultdict(int)
	for n in data['prof'].unique():
		dinner = data[data['prof'] == n]
		firstTime = min(dinner["timestamp"])
		for x in dinner['timestamp']:
			tdelta = ((x - firstTime).days * 24) + (x-firstTime).seconds/3600
			if tdelta > withinDay * 24:
				continue
			if not isinstance(tdelta, int):
				print tdelta
				print firstTime
				print x
				print n
			dict[tdelta] += 1
			
	if plot: 
		plt.bar(range(len(dict.keys())), dict.values())
		plt.xticks(range(len(dict.keys())), dict.keys())
		plt.show()
	return dict

#distribution of majors by certain dinners
def facultyMajorDist(data, num = 0, facList = [], plot = False):
	
	para = dictifierSpring2018()
	if num == 0 and len(facList) == 0:
		
		num = para.shape[0]
		list = para['FacultyKey']
	elif len(facList) == 0:
		list = para['FacultyKey'].head(num)
	else:
		list = facList
		num = len(facList)
	
	count = 1
	for x in list:
		plt.subplot(num, 1, count)

		facDist = data[data['prof'] == x]['major'].value_counts()
		facDist.plot(kind = 'bar')
		plt.title(x)
		plt.legend()
		
		
		#change labels
		locs, labs = plt.xticks()
		plt.xticks(locs, newLabels(labs))
		count += 1
	if plot:
		plt.tight_layout()
		plt.show()
	return facDist
	
def depMajorDist(department):
	facultyMajorDist(facList = departmentFinder(department)['FacultyKey'].values)

def facRelaterRating(para, data, G, faculty = "all"):

	facRatingDict = {}
	facRating = 0
	if faculty == "all":
		faculty = para['FacultyKey'].unique()
	paraSet = para.set_index('FacultyKey')
	fac = paraSet.loc[faculty]
	
	d = pd.merge(left = fac, right = data, left_on = 'FacultyKey', right_on = 'prof').set_index('prof')
	max = 0.
	for f in faculty:

		tempD = d.loc[f]
		for ind, x in tempD.iterrows():
			dep = [str(x['Department'])]
			if '|' in dep[0]:
				dep = dep[0].split('|')
			maj = x['major']
			#takes care of nan
			if isinstance(maj, float) or isinstance(dep, float):
				continue
				
			##### METHOD
			if count5Indict == 1:
				for n in dep:
					if G[n][maj]['count'] < 5:
						continue
			##### METHOD
			for n in dep:
				facRating += G[n][maj]['weight']
		m = float(facRating)
		if max < m:
			max = m
		facRatingDict[f] = m
	facRatingDict = {k: round(v/max, 2) for k, v in facRatingDict.iteritems()}
	
	return facRatingDict
	
#graph of relations
def relationsMajorGrapher():
	G = nx.Graph()

	para = dictifierSpring2018()
	para2 = dictifierFall2017()
	para3 = dictifierFall2018()

	d = pd.merge(right = para, left = dataSpring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = dataFall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = dataFall2018, right_on = 'FacultyKey', left_on = 'prof')
	
	da = pd.concat([d, d2, d3], sort = True)
	facD =  da['major'].value_counts()


	count = 0
	for index, x in da.iterrows():
		maj = str(x['major'])
		if '|' in maj:
			print "MULTIPLE: " + maj

		dep = [str(x['Department'])]
		if '|' in dep[0]:
			dep = dep[0].split('|')

		if maj == 'nan' or dep == 'nan':
			continue
		##### METHOD	
		if majIndict == 1:	
 			normalize = float(facD.loc[maj])
		else:
			normalize = 1.
		#####METHOD

		for n in dep:
			if G.has_edge(n, maj):
				G[n][maj]['weight'] += 1./normalize
				G[n][maj]['count'] +=1
				if G[n][maj]['count'] > 5:
					count +=1
			else:
				G.add_edge(n, maj, weight = 1./normalize)
				G.add_edge(n, maj, count = 1)
	return G

def attNormalizeRater(para, data, G):
	refSeries = data['prof'].value_counts()
	returnDict = {}
	
	total = float(sum(refSeries))
	dic = facRelaterRating(para, data, G)
	max = 0.
	for k, v in dic.iteritems():
		m = float(v)/float(refSeries.loc[k])
		if m > max:
			max = m
		returnDict[k] = m
	returnDict = {k: round(v/max, 2) for k, v in returnDict.iteritems()}
	return returnDict
	
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

def dinner_info(para, data, fac, plot_major = False):

	##DATA
	fADict, d = firstAcceptances(data)
	dinner = d[d['prof'] == fac]
	paraDin = para[para['FacultyKey'] == fac]
	G = relationsMajorGrapher()
	dep = [paraDin['Department'].values[0]]
	if '|' in dep[0]:
		dep = dep[0].split('|')
	arr = []
	for x in dep:
		arr.extend([(n, G[x][n]['weight']) for n in G.neighbors(x)])
	
	s = [(n[0], round(n[1], 2)) for n in sorted(arr, key = operator.itemgetter(1), reverse = True)[:5]]
	r = attNormalizeRater(para, data, G)
	facRelRat = facRelaterRating(para, data,G)


	##DATA
	
	print paraDin['FacultyName'].values[0]
	print "Department: " + paraDin['Department'].values[0].replace('|', ', ')
	print "Number of Students: " + str(len(dinner))
	print "Date: " + str(paraDin['Date'].values[0])
	print "Related Majors: " + str(s)
	print "Major-Diversity Rating (AN): " + str(r[fac])
	print "Major Diversity Rating: " + str(facRelRat[fac])
	print "First Acceptance Score: " + str(round(fADict[fac], 2))
		
 	depth = raw_input("depth?")
 	if depth == "y" or depth == "yes":
 		dinner = dinner.set_index('name')
 		print dinner[['major', 'firstAvalue']]
 		
 		
	if plot_major:
		plt.clf()
		plt.title(fac)
		dinner['major'].value_counts().plot.bar()
 		locs, labs = plt.xticks()
 		plt.xticks(locs, newLabels(labs))
		plt.show()


##WORKSPACE


indict = 1
if indict == 1:
	# print dataSpring2018.info()
	
	data = dataFall2018
	# print para.info()
# 	s = set(['bray', 'schott'])
	s = set(['astrachan'])
	
	para = dictifierFall2018()
	for n in para['FacultyKey']:
		if n in s:
			raw_input("next")
			dinner_info(para, data, n, plot_major = True)
			
if indict == 2:
	data = dataFall2018
	para = dictifierFall2018()
	
	G = relationsMajorGrapher()
# 	s =  sorted([(n, G[dep][n]['weight']) for n in G.neighbors(dep)], key = operator.itemgetter(1), reverse = True)
# 	print s
	frr = facRelaterRating(para, data, G)
	anr = attNormalizeRater(para, data, G)
	plt.hist([n[1] for n in dictSorter(frr, rev = "reverse")], bins = 25)
	plt.show()
	print dictSorter(frr, rev = "reverse")
	print dictSorter(anr, rev = "reverse")
# 	
# 	for n in sorted([(k, (frr[k] + anr[k])/2) for k in para['FacultyKey'].unique()], key = operator.itemgetter(1), reverse = True):
# 		print n[0] + '\t' + str(n[1])

	
	
if indict == 3:
	para = dictifierSpring2018()
	para2 = dictifierFall2017()
	para3 = dictifierFall2018()
 	data = dataFall2018

 	d = pd.merge(right = para, left = dataSpring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = dataFall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = dataFall2018, right_on = 'FacultyKey', left_on = 'prof')
	
	da = pd.concat([d3], sort = True)
	print "total applications = " + str(len(da))
	facD =  da.drop_duplicates('unique id')['major'].value_counts()
	facY =  da.drop_duplicates('unique id')['year'].value_counts()
	
	# facD.plot.bar()
	ti = set(dataFall2018['major'].unique())
	ti.update(dataSpring2018['major'].unique())
	x = dataSpring2018.drop_duplicates('unique id')['major'].value_counts()
	y = dataFall2018.drop_duplicates('unique id')['major'].value_counts()
	
	f = para3['Department'].value_counts()
	f2 = para['Department'].value_counts()
	dict = {}
	
	x = x/sum(x)
	y = y/sum(y)
	
	f = f/sum(f)
	f2 = f2/sum(f2)
	
	print f
	print f2

	normalize = 1.
	normalize2 = 1.
	for k in ti:
		if isinstance(k, float):
			continue
		if str(k) in f:
			
			normalize = f[k]
		if str(k) in f2:
			
			normalize2 = f2[k]
		try:
			
			dict[k] = (y[k]/normalize)-x[k]/normalize2
		except:
			try:
				dict[k] = y[k]/normalize
			except:
				try:
					dict[k] = -x[k]/normalize2
				except:
					continue
	for n in dictSorter(dict):
		print str(n[0]) + ": " + str(round(n[1], 2))
	dxl = pd.DataFrame.from_dict(dict.items())
	print dxl.describe()

	
	
# 	plt.show()
	
# 	propFRR = sum(facRelaterRating(para3, data, relationsMajorGrapher()).values())/len(data['prof'].unique())
# 	propANR = sum(attNormalizeRater(para3, data, relationsMajorGrapher()).values())/len(data['prof'].unique())
# 	print "propFRR = " + '%.2f' % propFRR
# 	print "propANR = " + '%.2f' % propANR
# 	print "total unique = " + str(sum(facD))
# 	print "FACD"
# 	print facD
# 	print "FACY"
# 	print facY
	
if indict == 4:
	lo = ""
	while lo != "end":
		lo = raw_input("student? ")
		if lo == "end":
			continue
		else:
			getStudent(lo)

if indict == 5:
	para = dictifierSpring2018()
	para2 = dictifierFall2017()
	para3 = dictifierFall2018()
 	data = dataFall2018

 	d = pd.merge(right = para, left = dataSpring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = dataFall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = dataFall2018, right_on = 'FacultyKey', left_on = 'prof')
	
	print para['Department'].value_counts()
	print para3['Department'].value_counts()
	print sum(para['Department'].value_counts())
	print sum(para3['Department'].value_counts())
##WORKSPACE









