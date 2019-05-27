import pandas as pd
import networkx as nx
from collections import Counter

data1 = pd.ExcelFile('Spring 2019 Potential Faculty Outreach.xlsx').parse(0)
dataSpring2019 = pd.ExcelFile('Spring 2019 Faculty Outreach.xlsx').parse(0)
dataFall2018 = pd.ExcelFile('Fall 2018 Faculty Outreach.xlsx').parse(0)

dataSpring2019["Name"] = dataSpring2019.First + " " + dataSpring2019.Last

def initialize():
	files = ['faculty1.txt']
	returnArray = {}
	for f in files:
		with open(f, "r") as current_file:
			array = current_file.readlines()
		for n in range(len(array)):
			tempArray = []
			if array[n] == "Contact":
				
		

def differenceSets2019():
	d1 = set([row['First_Name'] +" "+ row['Last_Name'] for index, row in dataSpring2019.iterrows()])
	d2 = set([row['First'] +" "+ row['Last'] for index, row in data2.iterrows()])

	r = set()
	for n in d2:
		if n in d1:
			r.add(n)

	print d1.difference(r)
	print d2.difference(r)

def countMajors(data):
	c = [d for n in data['Department'] for d in n.split(', ')]
	return Counter(c)
s = set(dataSpring2019["Name"])
r = set(dataFall2018["Name"])

dataSpring2019['Year'] = "Spring2019"
dataFall2018['Year'] = "Fall2018"
d = pd.concat([dataSpring2019, dataFall2018])

d = d.drop_duplicates('Name')

print countMajors(d)

initialize()




