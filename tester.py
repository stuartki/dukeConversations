from dukeC3 import init, dinner_info, relations_major_grapher
from time_eval import time_summary
from rater import fac_relate_rating, att_normalize_rater
import networkx as nx


d = init()
print(dinner_info(d['spring2018'][0], d['spring2018'][1], 'adair', plot_major=True))

indict = -20
if indict == 1:
	# print data_spring2018.info()
	
	data = data_fall2018
	# print para.info()
# 	s = set(['bray', 'schott'])
	s = set(['astrachan'])
	
	para = dictifierFall2018()
	for n in para['FacultyKey']:
		if n in s:
			raw_input("next")
			dinner_info(para, data, n, plot_major = True)
			
if indict == 2:
	data = data_fall2018
	para = dictifierFall2018()
	
	G = relations_major_grapher()
# 	s =  sorted([(n, G[dep][n]['weight']) for n in G.neighbors(dep)], key = operator.itemgetter(1), reverse = True)
# 	print s
	frr = fac_relate_rating(para, data, G)
	anr = att_normalize_rater(para, data, G)
	plt.hist([n[1] for n in dict_sorter(frr, rev = "reverse")], bins = 25)
	plt.show()
	print(dict_sorter(frr, rev = "reverse"))
	print(dict_sorter(anr, rev = "reverse"))
	
	dxl = pd.DataFrame.from_dict(anr.items())
	print(dxl.describe())
# 	
# 	for n in sorted([(k, (frr[k] + anr[k])/2) for k in para['FacultyKey'].unique()], key = operator.itemgetter(1), reverse = True):
# 		print n[0] + '\t' + str(n[1])

	
	
if indict == 3:
	para = dictifierSpring2018()
	para2 = dictifierFall2017()
	para3 = dictifierFall2018()
	data = data_fall2018

	d = pd.merge(right = para, left = data_spring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = data_fall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = data_fall2018, right_on = 'FacultyKey', left_on = 'prof')
	
	da = pd.concat([d3], sort = True)
	print("total applications = " + str(len(da)))
	facD =  da.drop_duplicates('unique id')['major'].value_counts()
	facY =  da.drop_duplicates('unique id')['year'].value_counts()
	
	# facD.plot.bar()
	ti = set(data_fall2018['major'].unique())
	ti.update(data_spring2018['major'].unique())
	x = data_spring2018.drop_duplicates('unique id')['major'].value_counts()
	y = data_fall2018.drop_duplicates('unique id')['major'].value_counts()
	
	f = para3['Department'].value_counts()
	f2 = para['Department'].value_counts()
	dict = {}
	
	x = x/sum(x)
	y = y/sum(y)
	
	f = f/sum(f)
	f2 = f2/sum(f2)
	
	print(f)
	print(f2)

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
	for n in dict_sorter(dict):
		print(str(n[0]) + ": " + str(round(n[1], 2)))
	dxl = pd.DataFrame.from_dict(dict.items())
	print(dxl.describe())

	
	
# 	plt.show()
	
# 	propFRR = sum(fac_relate_rating(para3, data, relations_major_grapher()).values())/len(data['prof'].unique())
# 	propANR = sum(att_normalize_rater(para3, data, relations_major_grapher()).values())/len(data['prof'].unique())
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
	data = data_fall2018

	d = pd.merge(right = para, left = data_spring2018, right_on = 'FacultyKey', left_on = 'prof')
	d2 = pd.merge(right = para2, left = data_fall2017, right_on = 'FacultyKey', left_on = 'prof')
	d3 = pd.merge(right = para3, left = data_fall2018, right_on = 'FacultyKey', left_on = 'prof')
	
	# print para['Department'].value_counts()
	# print para3['Department'].value_counts()
	# print sum(para['Department'].value_counts())
	# print sum(para3['Department'].value_counts())
	
if indict == 6:
	data = data_fall2018
	para = dictifierFall2018()
	
	G = relations_major_grapher()
	
	dic = {}
	for fac in data['prof'].unique():
		print(fac)
		if len(data[data['prof'] == fac]) > 8:
			continue
		dep = para.set_index('FacultyKey').loc[fac, "Department"].split('|')[0]
		n = len(data[data["prof"] == fac])
		score = fac_relate_rating(para, data, G, norm = False)[fac]
		dic[fac] = probFinder(G, dep, n, score)
	dxl = pd.DataFrame.from_dict(dic.items())
	dxl.to_csv("tempProb.csv")
	print(dxl.head())
	print(dxl.describe())

if indict == 7:
	textAnalysis()
	

##WORKSPACE









