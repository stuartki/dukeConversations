#nice algorithms

import operator
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def dictSorter(dict, rev = ""):
	import operator
	if rev == "reverse":
		r = False
	else:
		r = True
	sorted_x = sorted(dict.items(), key=operator.itemgetter(1), reverse = r)
	return sorted_x	
	
def isSimilar(input, string, prop = 0):
	s = [n for n in string]
	if prop == 0:
		prop = abs(len(input) - len(string)) - 1
	count = 1
	for n in input:
		if n in s:
			
			count+=1
	if len(s) > prop and count + 1 > len(s):
		return True
	else:
		return False
			
		
def tfidfVect():
	texts = [
		"good movie", "not a good movie", "did not like", 
		"i like it", "good one"
	]
	# using default tokenizer in TfidfVectorizer
	tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
	features = tfidf.fit_transform(texts)
	d = pd.DataFrame(
		features.todense(),
		columns=tfidf.get_feature_names()
	)

	return d

def multinomial(xs, ps):
	from numpy import array, log, exp
	from scipy.special import gammaln
	def log_factorial(x):
		return gammaln(array(x)+1)
	n = sum(xs)
	xs, ps = array(xs), array(ps)
	result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
	return exp(result)
		
		