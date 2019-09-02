#!/usr/bin/env

from datetime import datetime
from itertools import combinations
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from string import punctuation
from tqdm import tqdm
import glob
import numpy as np
import random
import re
import scipy

"""
Paths 
-----
Give paths to directories for dev, impostors and test
Also paths where results are written to.
"""

impostors_dir = 'abelard-heloise/imp_set'
train_dir = 'abelard-heloise/train_set'
test_dir = 'abelard-heloise/test_set'

# 'a' parameter stands for 'append'
results_file = open('abelard-heloise/output/results-15-06-19.txt', 'a')
evaluations_file = open('abelard-heloise/output/evaluations-15-06-19.txt', 'a')

"""
Parameters
----------
"""

k = 100
sample_len = 650
step_size = 100

"""
Functions 
---------
"""
def load_and_split(path):
	"""
	Function that opens document and preprocesses it.
		- Removal of punctuation
		- Removal of uppercase
		- Sampling (divide into samples)
	
	Parameters
	----------
	path: directory path
	
	Returns
	-------
	d = {author = [(sample name, '...'),
		   	   	   (sample name, '...'),
				   (...)]
	}
	
	"""
	
	d = {}
	for fn in glob.glob(path + '/*'):
		author = fn.split('/')[-1].split('_')[0]
		d[author] = []
	
	# Loop over directory and collect as training set
	for fn in glob.glob(path + '/*'):
		author = fn.split('/')[-1].split('_')[0]
		title = fn.split('/')[-1].split('_')[1].split('.')[0]
		text = open(fn).read()
		# Escape punctuation and make characters lowercase
		text = re.sub('[%s]' % re.escape(punctuation), '', text)
		# Escape digits
		text = re.sub('\d+', '', text)
		text = text.lower().split()
		# Sampling of the text
		text_samples = [text[i:i+sample_len] for i in range(0, len(text), sample_len)]
		text_samples = [i for i in text_samples if len(i) == sample_len]
		# Append to training set
		for idx, sample in enumerate(text_samples):
			sample = ' '.join(sample)
			sample_title = '{}_{}'.format(title, str(idx))
			d[author].append((sample_title, sample))

	return d

def scramble_and_vectorize(x_heldout, x_impostors, features, subset_size):
	"""
	Parameters
	----------
	Returns
	-------
	"""
	# Feature subset was fixed when processing whole corpus
	feature_subset = random.sample(features, subset_size)
	vectorizer = TfidfVectorizer(vocabulary=feature_subset)
	
	x_vec = vectorizer.fit_transform([x_heldout]).toarray()
	x_candidates = vectorizer.transform(x_impostors).toarray()
	
	# array.reshape(1, -1) if it contains a single sample
	scaler = StandardScaler()
	x_candidates = scaler.fit_transform(x_candidates)
	x_vec = scaler.transform(x_vec.reshape(1,-1))

	return x_vec, x_candidates

def scorer(metric, tp, tn, fp, fn):
	try: 
		if metric == 'accuracy':
			score = (tp + tn) / (tp + tn + fp + fn)
		elif metric == 'precision':
			score = tp / (tp + fp)
		elif metric == 'recall':
			score = tp / (tp + fn)
		elif metric == 'fpr':
			score = fp / (fp + tn)
		score = np.round(score, decimals=2)
		return score
	except ZeroDivisionError:
		return 0

def impostors_method(data, all_train_data, impostors, n_impostors, same_author, impostor_sampling):
	"""
	Parameters
	----------
	data: 					dictionary {author: [(title, sample), ...]} 
	all_train_data: 		dictionary {author: [(title, sample), ...]} of development set 
	impostors: 				dictionary {author: [(title, sample), ...]} of impostors set
	samples_per_a:			takes an integer number of samples per author; overrules n_impostors
	n_impostors: 			number of randomly sampled impostors
	same-author: 			include <same-author>, 'y' (yes) or 'n' (no)
	
	Returns 
	-------
	confidences_per_class: 	list of percentages indicating confidence of impostors algorithm
	matches: 				list of tuples containing gold truth label '<diff/same-authorship>' 
							and prediction '<diff/same-authorship>'
	"""
	confidences = []
	matches = []
	ground_truths = []
	for y_heldout, titles_and_samples in data.items():
		# Iterate over samples in input data
		for (title, x_heldout) in tqdm(titles_and_samples, total=len(titles_and_samples), postfix='sample'):
			# Dictionary to count predictions per label
			predictions = {i: 0 for i in impostors}
			for i in all_train_data:
				predictions[i] = 0

			# Random impostors
			if same_author == 'y':
				# Include n training text that is <same-author>
				# Only cross-text
				pool = [tup for tup in all_train_data[y_heldout] if tup[0].split('_')[0] != title.split('_')[0]]			
				random_tup = random.choice(pool)
				random_samp_title = random_tup[0]
				random_sample = random_tup[1]
				selected_sample = [(y_heldout, random_sample)]
			elif same_author == 'n':
				# Include 1 training text that is <diff-author>
				pool = [i for i in list(all_train_data.keys()) if i != y_heldout]
				random_diff_author = random.choice(pool)
				random_tup = random.choice(all_train_data[random_diff_author])
				selected_sample = [(random_diff_author, random_tup[1])]
			elif same_author == 'a':
				# 'a' stands for all candidates: all dev candidates are included with impostors
				# Usually only in final test phase
				pool = list(all_train_data.keys())
				selected_sample = []
				for author in pool:
					random_tup = random.choice(all_train_data[author])
					selected_sample.append((author, random_tup[1]))

			# Adjust number of impostors dependent on how many authors from training 
			# set are introduced above
			adjusted_n = n_impostors - len(selected_sample)
			
			# Start k=100 iterations
			for iteration in tqdm(range(0,k), postfix='k iterations'):

				if impostor_sampling == 'equal':
					random_impostors = []
					for author_key in impostors.keys():
						random_tup = random.choice(impostors[author_key])
						random_impostors.append((author_key, random_tup[1]))
					for i in range(0,len(selected_sample)):
						# Pop n random impostors from list so to keep number of impostors equal
						random_impostors.pop(random.randrange(len(random_impostors)))
					random_impostors = random_impostors + selected_sample

				elif impostor_sampling == 'random':
					random_impostors = []
					for _ in range(0, adjusted_n):
						random_author = random.choice(list(impostors.keys()))
						random_tup = random.choice(impostors[random_author])
						random_impostors.append((random_author, random_tup[1]))
					random_impostors = random_impostors + selected_sample

				x_impostors = [sample for (author, sample) in random_impostors]
				y_impostors = [author for (author, sample) in random_impostors]
				
				x_vec, x_candidates = scramble_and_vectorize(x_heldout, x_impostors, features, subset_size)

				cosine_similarities = [np.abs(cosine_similarity(x_vec, [i])) for i in x_candidates]
				cosine_similarities = [np.ndarray.flatten(i)[0] for i in cosine_similarities]

				argmax_idx = np.argmax(cosine_similarities)
				y_pred = y_impostors[argmax_idx]
				predictions[y_pred] += 1

			# Compute confidence of attribution over k loops for each of the candidates
			scores = [[y, x/100] for y, x in predictions.items()]
			scores.sort(key=lambda x:x[1], reverse=True)
			candidates = [i[0] for i in scores]
			confidence_rates = [i[1] for i in scores]
			y_candidate = candidates[0]
			confidence = confidence_rates[0]

			results_file.write(y_heldout + ' // ')
			for cand in candidates: 
				results_file.write(cand + ' ')
			results_file.write('\n')
			for score in confidence_rates:
				results_file.write(str(score) + ' ')
			results_file.write('\n')

			if same_author == 'y':
				ground_truth = '<same-author>'
			elif same_author == 'n':
				ground_truth = '<diff-author>'
			elif same_author == 'a':
				ground_truth = str(title.split('_')[0])

			ground_truths.append(ground_truth)
			confidences.append(confidence)
			matches.append((y_heldout, y_candidate))

	return confidences, matches, ground_truths

def evaluate_settings(y_probas, y_true, input_sigma):
	"""
	Parameters
	----------
	y_probas: list of percentages indicating confidence of impostors algorithm
	y_true: list of labels
	matches: list of tuples containing predictions
	input_sigma: return scores for a particular sigma value equal or lower than input sigma
	
	Returns
	-------
	_accuracy_: accuracy for given sigma value(s)
	_precision_: precision for given sigma value(s)
	_recall_: recall for given sigma value(s)
	_sigma_: sigma
	"""

	le = LabelEncoder()
	y_true = le.fit_transform(y_true)
	precisions, recalls, sigmas = precision_recall_curve(y_true, y_probas)
	
	accuracies = []
	for sigma in sigmas:
		y_preds = []
		for conf in y_probas:
			if conf >= sigma:
				# if confidence greater than sigma, make prediction 1 (<same-author>)
				y_preds.append(1)
			else:
				# elif confidence smaller than sigma, make prediction 0 (<same-author>)
				y_preds.append(0)
		np.array(y_preds)
		accuracy = metrics.accuracy_score(y_true, y_preds)
		accuracies.append(accuracy)
	accuracies = np.array(accuracies)
	sigmas = np.round(sigmas, decimals=2)
	accuracies = np.round(accuracies, decimals=2)
	precisions = np.round(precisions, decimals=2)
	recalls = np.round(recalls, decimals=2)

	evaluations_file.write('\n' + 'sigma' + '\t' + 'accuracy' + '\t' + 'precision' + '\t' + 'recall' + '\n')
	
	if input_sigma == None:
		# this will be the case during training
		# simply yield all sigma values
		for (sigma, accuracy, precision, recall) in zip(sigmas, accuracies, precisions, recalls):
			evaluations_file.write(str(sigma) + '\t' +
								   str(accuracy) + '\t' +
								   str(precision) + '\t' +
								   str(recall) + '\n')
		idx = np.argmax(accuracies)
	
	else:
		# Mostly the case during testing, when only result of given sigma is called for
		if input_sigma in sigmas:
			# equal sigma results
			idx = list(sigmas).index(input_sigma)
		else:
			# previous smaller sigma results
			temp_list = list(sigmas)
			temp_list.append(input_sigma)
			temp_list = sorted(temp_list)
			idx = temp_list.index(input_sigma)-1

		evaluations_file.write(str(sigmas[idx]) + '\t' +
					   str(accuracies[idx]) + '\t' +
					   str(precisions[idx]) + '\t' +
					   str(recalls[idx]) + '\n')

	_accuracy_, _precision_, _recall_, _sigma_ = accuracies[idx], precisions[idx], recalls[idx], sigmas[idx]

	return _accuracy_, _precision_, _recall_, _sigma_

"""
Load data
---------
"""

all_train_data = load_and_split(train_dir)
prelim_test_data = load_and_split(test_dir)
impostors_data = load_and_split(impostors_dir)

"""
Feature extraction 
------------------
Extract a full feature set (the entire vocabulary)
Join entire corpus and make feature set
of the dev set (the impostors!).
"""
# Collects all samples from dev, test and impostors

all_train_samples = [i[1] for i in sum(list(all_train_data.values()), [])]
test_samples = [i[1] for i in sum(list(prelim_test_data.values()), [])]
impostors_samples = [i[1] for i in sum(list(impostors_data.values()), [])]

entire_corpus = all_train_samples + test_samples + impostors_samples
entire_corpus = ' '.join(entire_corpus)

features = {}
model = TfidfVectorizer(analyzer='word')
array = model.fit_transform([entire_corpus]).toarray()
array = np.ndarray.flatten(array)
features = model.get_feature_names()

subset_size = int(np.around(len(features)/2))

"""
Data management
---------------

Training data
Dev and heldout
"""
# n samples is split in dev and test
n_samples = 40
train_data_partition = {}
for author, titles_and_samples in all_train_data.items():
	selection = random.sample(titles_and_samples, n_samples)
	train_data_partition[author] = selection

dev_data = {}
heldout_data = {}
for author, titles_and_samples in train_data_partition.items():
	dev_split, heldout_split = train_test_split(titles_and_samples)
	dev_data[author] = dev_split
	heldout_data[author] = heldout_split

"""
Test data
---------
"""
dir_location = 'abelard-heloise/test_set'
test_data = load_and_split(dir_location)

"""
shingling
uncheck for shingling
"""
test_set_location = '/Users/jedgusse/compstyl/impostors-method/abelard-heloise/test_set/Anon_letter-collection-complete.txt'
original_test_txt = open(test_set_location).read()
test_txt = re.sub('[%s]' % re.escape(punctuation), '', original_test_txt) # Escape punctuation and make characters lowercase
text_txt = re.sub('\d+', '', test_txt)
test_txt = test_txt.lower().split()

# Collect test samples in dictionary by keyword of range
test_data = {}
# "shingling": make windows	
steps = np.arange(0, len(test_txt), step_size)

step_ranges = []
test_data = {}
key = dir_location.split('/')[-1].split('.')[0].split('_')[0]
test_data[key] = []
for each_begin in steps:
	sample_range = range(each_begin, each_begin + sample_len)
	step_ranges.append(sample_range)
	text_sample = []
	for index, word in enumerate(test_txt):
		if index in sample_range:
			text_sample.append(word)
	title = '{}-{}'.format(str(each_begin), str(each_begin + sample_len))
	if len(text_sample) == sample_len:
		value = (title, ' '.join(text_sample))
		test_data[key].append(value)

"""
Development
-----------
Train on dev set
"""

results_file.write('Development set results \n')
results_file.write('----------------------- \n\n')
# Train <same-author> and <diff-author> pairs
results_file.write('<same-author> training \n\n')
dev_same_confs, dev_same_matches, dev_same_ground = impostors_method(dev_data, all_train_data, impostors_data, n_impostors=20, same_author='y', impostor_sampling='random')
results_file.write('\n<diff-author> training \n\n')
dev_diff_confs, dev_diff_matches, dev_diff_ground = impostors_method(dev_data, all_train_data, impostors_data, n_impostors=20, same_author='n', impostor_sampling='random')
results_file.write('----------------------- \n\n')

evaluations_file.write('Development set evaluation \n')
evaluations_file.write('----------------------- \n\n')
# Make evaluation and find suitable sigma
dev_y_probas = dev_same_confs + dev_diff_confs
dev_y_true = dev_same_ground + dev_diff_ground
dev_acc, dev_precision, dev_recall, dev_sigma = evaluate_settings(dev_y_probas, dev_y_true, input_sigma=None)
evaluations_file.write('----------------------- \n\n')

"""
Heldout
-------
Test on heldout set
"""

results_file.write('Heldout set results \n')
results_file.write('----------------------- \n\n')
# Test dev sigma settings on <same-author> and <diff-author> pairs
results_file.write('<same-author> heldout \n\n')
heldout_same_confs, heldout_same_matches, heldout_same_ground = impostors_method(heldout_data, all_train_data, impostors_data, n_impostors=20, same_author='y', impostor_sampling='random')
results_file.write('\n<diff-author> heldout \n\n')
heldout_diff_confs, heldout_diff_matches, heldout_diff_ground = impostors_method(heldout_data, all_train_data, impostors_data, n_impostors=20, same_author='n', impostor_sampling='random')
results_file.write('----------------------- \n\n')

evaluations_file.write('Heldout set evaluation \n')
evaluations_file.write('----------------------- \n\n')
# Make evaluation and with predefined dev sigma
heldout_y_probas = heldout_same_confs + heldout_diff_confs
heldout_y_true = heldout_same_ground + heldout_diff_ground
heldout_acc, heldout_precision, heldout_recall, heldout_sigma = evaluate_settings(heldout_y_probas, heldout_y_true, input_sigma=dev_sigma)
evaluations_file.write('----------------------- \n\n')

"""
Test
-------
Test on actual test set
"""

results_file.write('Test set results \n')
results_file.write('----------------------- \n\n')
test_confs, test_matches, test_ground = impostors_method(test_data, all_train_data, impostors_data, n_impostors=20, same_author='a', impostor_sampling='random')
results_file.write('----------------------- \n\n')

"""
Closing experiment, writing out files
-------------------------------------
"""
# Prints date and time of experiment
n = datetime.now()
time = '%s'%n

results_file.write(time + '\n\n\n')
evaluations_file.write(time + '\n\n\n')

results_file.close()
evaluations_file.close()