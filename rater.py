#to create a self-initialized graph of edges between FACULTY 'Department' and STUDENT 'major'
#weighted edges of how many connections over total_data
#standardized rating over 1 for both normalized per dinner attendance and not

#returns
#fac_relate_rating - dictionary of professor scores by semester based on graph
#att_normalize_rater - dictionary of professor scores by semester normalized by dinner attendance
#first_acceptances - dictionary of professor scores by semester for professors who brought in new dinner attendees

import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

##rating INDICTS
maj_indict = 1
att_indict = 1
count_indict = 1

#rating system to add scores from graph of dinner participants
#an individual has higher weight if a pairing is UNLIKELY
def fac_relate_rating(data, para, G, faculty = "all", norm = True):

	facRatingDict = {}
	facRating = 0
	if faculty == "all":
		faculty = para['FacultyKey'].unique()
	paraSet = para.set_index('FacultyKey')
	fac = paraSet.loc[faculty]
	
	
	d = pd.merge(left = fac, right = data, left_on = 'FacultyKey', right_on = 'prof').set_index('prof')
	max = 0.
	for f in faculty:
		facRating = 0
		tempD = d.loc[f]
		for ind, x in tempD.iterrows():
			dep = [str(x['Department'])]
			if '|' in dep[0]:
				dep = dep[0].split('|')
			maj = x['major']
			#takes care of nan
			if isinstance(maj, float) or isinstance(dep, float):
				continue
				
			##### don't include low instance counts, will skew numbers
			if count_indict == 1:
				for n in dep:
					if G[n][maj]['count'] < 5:
						continue
			##### METHOD

			#add weight for each pairing
			for n in dep:
				facRating += G[n][maj]['weight']
		m = float(facRating)
		if max < m:
			max = m
		facRatingDict[f] = m
	if not norm:
		max = 1		
	
	facRatingDict = {k: round(v/max, 2) for k, v in facRatingDict.items()}
	
	return facRatingDict


#normalize by attendance per dinner
def att_normalize_rater(data, para, G):
	returnDict = {}

	refSeries = data['prof'].value_counts()
	total = float(sum(refSeries))

	frr = fac_relate_rating(data, para, G)
	returnDict = {k:(float(v)/float(refSeries.loc[k]))*total for k, v in frr.items()}
	m = max(returnDict.values())
	returnDict = {k: round(v/m, 2) for k, v in returnDict.items()}
	return returnDict
	
#get score of which professor brings in the most first-time attendees
def first_acceptances(data, plot = False):

	ref_series = data['prof'].value_counts()
	student_count = data['unique id'].value_counts()

	#organize data by timestamp to be able to tell first attendee
	data = data.sort_values(by = ['timestamp'])
	data = data.reset_index()
	
	#creates new column normalized by how many attendances
	firstAvalue = pd.Series(np.ones(len(data)), index=data.index)
	for index, row in data.iterrows():
		firstAvalue[index] = 1./float(student_count[row['unique id']])
	data['firstAvalue'] = firstAvalue

	#add up scores normalized by the attendance of the professor
	fA = defaultdict(int)
	for index, row in data.iterrows():
		fA[row['prof']] += row['firstAvalue']/ref_series[row['prof']]

	if plot:
		pd.Series(fA).plot.bar()
		plt.tight_layout()
		plt.show()
	#also returns data with firstAvalue column
	# a score of 1 would be perfect first attendance AND an attendance of dinner of 1
	# not as "valuable" if student has applied many times
	return fA, data