#for the purpose of evaluating the effect of the time of application
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

#distribution of times by dinner
def time_dinner(data, fac = ""):
	# ti = data['timestamp']
    #allow toggle through each dinner's time summary
	if fac == "":
		for n in data['prof'].unique():
			input(n)
			dinner = data[data['prof'] == n]
			dinner.groupby(dinner["timestamp"].dt.hour)["timestamp"].count().plot(kind="bar")
			plt.title(n + ": First Time = " + str(min(dinner["timestamp"])))
			plt.show()
    #specific dinner time summary
	else:
		dinner = data[data['prof'] == fac]
		dinner.groupby(dinner["timestamp"].dt.hour)["timestamp"].count().plot(kind="bar")
		plt.title(fac)
		plt.show()

#by hour, distribution of time of applications GIVEN the first_time of each dinner
def time_summary(data, plot = False, withinDay = 1):
	dict = defaultdict(int)
	for n in data['prof'].unique():
		dinner = data[data['prof'] == n]
		first_time = min(dinner["timestamp"])
		for x in dinner['timestamp']:
			tdelta = int(((x - first_time).days * 24) + (x-first_time).seconds/3600)
			if tdelta > withinDay * 24:
				continue
			dict[tdelta] += 1
	
	if plot:
		plot_array = [dict[n] if n in dict.keys() else 0 for n in range(24*withinDay)]
		plt.bar(range(24*withinDay), plot_array)
		plt.xticks(range(24*withinDay), np.array(range(24*withinDay))+ 1)
		plt.show()

	return dict