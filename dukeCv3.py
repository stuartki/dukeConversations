#general methods for analyzing duke conversations data
#relations_major_grapher - graph of edges between department and majors
#dinner_info
import csv
import operator
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from itertools import islice, combinations, combinations_with_replacement
import math
import networkx as nx
from algorithms import isSimilar
import math
import time
import operator as op
from functools import reduce

from rater import fac_relate_rating, att_normalize_rater, first_acceptances

#returns dictionary of data and parameters
def init():
    path = '/Users/stuartki/Dropbox/dukeConversations/'

    data_spring2018 = pd.ExcelFile('spring2018.xlsx').parse(0)
    data_fall2017 = pd.ExcelFile('fall2017.xlsx').parse(0)
    data_fall2018 = pd.ExcelFile('fall2018.xlsx').parse(0)
    total_data = pd.concat([data_spring2018, data_fall2017, data_fall2018], sort = True)

    #returns dinner parameters
    para_spring2018 = pd.ExcelFile(path + 'Spring2018Parameters.xlsx').parse(0)
    para_fall2017 = pd.ExcelFile(path + 'Fall2017Parameters.xlsx').parse(0)
    para_fall2018 = pd.ExcelFile(path + 'Fall2018Parameters.xlsx').parse(0)

    return {"spring2018": (data_spring2018, para_spring2018),
            "fall2017": (data_fall2017, para_fall2017),
            "fall2018": (data_fall2018, para_fall2018),
            "total": total_data}

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer/denom

#cleans duplicates in certain col_name - USE ONCE
def cleaner(data, col_name):
	re = data.drop_duplicates(subset = col_name, keep = 'first')
	re.to_csv('fall2017test.csv', index = False)

#shorter label markers
def labeler(data):
	majors = set(data['major'].unique())
	dict = {}
	for x in majors:
		if x == "Biomedical Engineering":
			dict[x] = "BME"
			continue
		dict[x] = str(x)[:2]
	return dict

#labels for nan
def new_labels(labs, data):
	newLabs = [labeler(data)[x.get_text()] for x in labs if x.get_text() !='nan']
	newLabs.append('nan')
	return newLabs

#sort dictionary by values, or keys given index
def dict_sorter(x, rev = "", index = 1):
	if rev == "reverse":
		r = True
	else:
		r = False
	sorted_x = sorted(x.items(), key=operator.itemgetter(index), reverse = r)
	return sorted_x
	
#distribution of all majors
def major_grapher(data):
	data['major'].value_counts().plot(kind = 'bar', fontsize = 10)

	#plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	#annoying label changer
	locs, labs = plt.xticks()
	plt.xticks(locs, new_labels(labs, data))
	plt.tight_layout()
	plt.show()
	
#return professors of "x" department
def department_finder(parameters, dep):

	parameters['strsplit'] = parameters.Department.str.split("|")
	parameters['trueFalse'] = parameters['strsplit'].apply(lambda x: 'true' if dep in x else 'false') == 'true'
	profsDep = parameters[parameters['trueFalse']]
	return profsDep

#distribution of majors by certain dinners
def faculty_major_dist(data, para, num = 0, fac_list = [], plot = False):
	#initialize num and fac_list
	if num == 0 and len(fac_list) == 0:
		num = para.shape[0]
		list = para['FacultyKey']
	elif len(fac_list) == 0:
		list = para['FacultyKey'].head(num)
	else:
		list = fac_list
		num = len(fac_list)
	
	count = 1
	#plot for each faculty in list
	for x in list:
		plt.subplot(num, 1, count)

		facDist = data[data['prof'] == x]['major'].value_counts()
		facDist.plot(kind = 'bar')
		plt.title(x)
		plt.legend()
		#change labels
		locs, labs = plt.xticks()
		plt.xticks(locs, new_labels(labs, data))
		count += 1
	if plot:
		plt.tight_layout()
		plt.show()
	#return the list of major counts
	return facDist
	
#find all faculty major distributions of a department
def dep_major_dist(data, para, department):
	faculty_major_dist(data, para, fac_list = department_finder(para, department)['FacultyKey'].values)

#graph of relations between FACULTY 'Department' and STUDENT 'major'
def relations_major_grapher():
	G = nx.Graph()
	initial = init()
	data_spring2018 = initial['spring2018'][0]
	data_fall2017 = initial['fall2017'][0]
	data_fall2018 = initial['fall2018'][0]
	para = initial['spring2018'][1]
	para2 = initial['fall2017'][1]
	para3 = initial['fall2018'][1]

    #merge the two datasets to combine faculty and student data
	d = pd.merge(right = para, left = data_spring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = data_fall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = data_fall2018, right_on = 'FacultyKey', left_on = 'prof')
	
    #total data for all semesters
	da = pd.concat([d, d2, d3], sort = True)
	facD =  da['major'].value_counts()
	

	count = 0
	for index, x in da.iterrows():
        #get major and department of student and faculty
		maj = str(x['major'])
		if '|' in maj:
            #check if this input system is still in play
			print("MULTIPLE: " + maj)

		dep = [str(x['Department'])]
		if '|' in dep[0]:
			dep = dep[0].split('|')
		if maj == 'nan' or dep == 'nan':
			continue
        
		##### normalize over the total number of majors in the whole dataset	
		#an individual has higher weight if a pairing is UNLIKELY
		normalize = float(facD.loc[maj])


		#populate the undirected graph
		for n in dep:
			if G.has_edge(n, maj):
				G[n][maj]['weight'] += 1./normalize
				G[n][maj]['count'] +=1
				if G[n][maj]['count'] > 5:
					#count variable just to test
					count +=1
			else:
				G.add_edge(n, maj, weight = 1./normalize)
				G.add_edge(n, maj, count = 1)
	return G



#summary of data
def dinner_info(data, para, fac, plot_major = False):

	##DATA
	fADict, d = first_acceptances(data)

	dinner = d[d['prof'] == fac]
	paraDin = para[para['FacultyKey'] == fac]
	G = relations_major_grapher()

	#get "related majors" from relations graph
	dep = [paraDin['Department'].values[0]]
	if '|' in dep[0]:
		dep = dep[0].split('|')
	arr = []
	for x in dep:
		arr.extend([(n, G[x][n]['weight']) for n in G.neighbors(x)])
	s = [(n[0], round(n[1], 2)) for n in sorted(arr, key = operator.itemgetter(1), reverse = True)[:5]]
	
	
	r = att_normalize_rater(data, para, G)
	facRelRat = fac_relate_rating(data, para,G)


	##DATA
	
	print(paraDin['FacultyName'].values[0])
	print("Department: " + paraDin['Department'].values[0].replace('|', ', '))
	print("Number of Students: " + str(len(dinner)))
	print("Date: " + str(paraDin['Date'].values[0]))
	print("Related Majors: " + str(s))
	print("Major-Diversity Rating (AN): " + str(r[fac]))
	print("Major Diversity Rating: " + str(facRelRat[fac]))
	print("First Acceptance Score: " + str(round(fADict[fac], 2)))   

  # depth = raw_input("depth?")
  # if depth == "y" or depth == "yes":
 	# 	dinner = dinner.set_index('name')
 	# 	print(dinner[['major', 'firstAvalue']])
 		
	if plot_major:
		plt.clf()
		plt.title(fac)
		dinner['major'].value_counts().plot.bar()
		locs, labs = plt.xticks()
		plt.xticks(locs, new_labels(labs, data))
		plt.show()

#needs to be shortened, more efficient
#STILL IN WORK
def probFinder_v1(G, dep, n, score):
	count = 0
	totcount = 0
	timeMinus = 0.
	de = 0.
	m = len(G.edges(dep))
	total = float(float(math.factorial(m + n - 1.))/(float(math.factorial(n))*float(math.factorial(m-1))))

	trueStart = time.time()

	for c in combinations_with_replacement(G.edges(dep), n):
		curScore = 0.
		start = time.time()
		for e in c:
			curScore += G[e[0]][e[1]]['weight']
		if curScore > score:
			count+=1
		totcount +=1
		end = time.time()
		de = (de*float(totcount-1) + end - start)/float(totcount)
		timeMinus = end - trueStart
		if totcount % 1000000 == 0:
			print("TIME TAKEN = " + str(round(timeMinus/60., 2)) + " minutes")
			print("TIME LEFT = " + str(round(de * total/60. - timeMinus/60., 2)) + " minutes")
	print("PROBABILITY: Score > " + str(score) + " = " + str(float(count)/float(totcount)))
	return float(count)/float(totcount)

#STILL NEEDS WORK

	



