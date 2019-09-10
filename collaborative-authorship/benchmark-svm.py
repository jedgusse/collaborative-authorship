#!/usr/bin/env

from binascii import hexlify
from collections import Counter
from itertools import combinations, compress
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from preprocess import DataReader, Vectorizer
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_regression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, LabelBinarizer
from statsmodels.nonparametric.smoothers_lowess import lowess
from string import punctuation
from tqdm import tqdm
from tqdm import trange
import argparse
import colorsys
import glob
import heapq
import itertools
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pickle
import random
import re
import seaborn.apionly as sns
import sys
import warnings

# Filters out warning of Data conversion, comes with standard scaling
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')

# Load files
train_location = '/Users/jedgusse/compstyl/impostors-method/abelard-heloise/train_set'

"""
FUNCTION TO MAKE SPARSE MATRIX INTO DENSE MATRIX
------------------------------------------------
"""
# Intermediary step, enables all the NaN values to become equal to 0.
		# Vectorizer outputs sparse matrix X
		# This function returns X as a dense matrix

def to_dense(X):
		X = X.todense()
		X = np.nan_to_num(X)
		return X

def most_common(lst):
	return max(set(lst), key=lst.count)

"""
Load and Split function
-----------------------
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

"""
STYLOMETRY PARAMETERS
---------------------
"""
# All parameters that are declared a priori
# Potential vocabularies fed in to the vectorizers

step_size = 100
test_dict = {}
function_words_only = open('/Users/jedgusse/compstyl/params/fword_list.txt').read().split()
invalid_words = []
rnd_dct = {'n_samples': 140,
		   'smooth_train': True,
		   'smooth_test': False}

results_file = open('/Users/jedgusse/compstyl/output/numpy_output/abel-heloise-training.txt', 'a')

# Search parameters

"""settings for mdp"""
# feat_type_loop = ['raw_fwords', 'tfidf_fwords']
# s_ls = list(range(2000,5500,500))
# # feat_loop = list(range(250,1250,250))
# feat_loop = list(range(50,350,50))
# cs = [1, 10, 100, 1000]
# n_cval_splits = 20

"""settings for suger"""

feat_type_loop = ['raw_fwords', 'raw_MFW', 'raw_4grams', 'tfidf_fwords', 'tfidf_MFW', 'tfidf_4grams']
c_options = [1]
sample_size = 1200
feat_n_loop = [200, 400]
n_selected_feats = [50]
"""
Watch out: extremely low n_cval_splits; normal number would be 40
"""
n_cval_splits = 7
# randomization_params = [(True, False), (False, True)]
randomization_params = [(True, False)]

for feat_type in feat_type_loop:
	for n_feats in feat_n_loop:
		# Leave One Out cross-validation
		"""
		PREPROCESSING
		-------------
		"""
		# Load training files
		# The val_1 and val_2 pass True or False arguments to the sampling method
		for val_1, val_2 in randomization_params:
			train_authors, train_titles, train_texts = DataReader(train_location, sample_size,
												test_dict, rnd_dct
												).metadata(sampling=val_1,
												type='folder',
												randomization=val_2)
			
			# Try both stratified cross-validation as 'normal' KFold cross-validation.
			# Stratification has already taken place with random sampling
			cv_types = []
			if val_2 == False:
				sampling_type = 'normal'
				cv_types.append(StratifiedKFold(n_splits=n_cval_splits))
				# cv_types.append(KFold(n_splits=n_cval_splits))
			elif val_2 == True:
				sampling_type = 'random'
				cv_types.append(KFold(n_splits=n_cval_splits))

			# Number of splits is based on number of samples, so only possible afterwards
			# Minimum is 2
			# n_cval_splits = list(range(2,len(train_texts)+1))

			"""
			ACTIVATE VECTORIZER
			"""
			if feat_type == 'raw_MFW': 
				vectorizer = CountVectorizer(stop_words=invalid_words, 
										   analyzer='word', 
										   ngram_range=(1, 1),
										   max_features=n_feats)
			elif feat_type == 'tfidf_MFW': 
				vectorizer = TfidfVectorizer(stop_words=invalid_words, 
										   analyzer='word', 
										   ngram_range=(1, 1),
										   max_features=n_feats)
			elif feat_type == 'raw_fwords':
				"""
				The entire corpus is searched and all words 
				that are not function words are rendered invalid
				"""
				stop_words = [t.split() for t in train_texts]
				stop_words = sum(stop_words, [])
				stop_words = [w for w in stop_words if w not in function_words_only]
				stop_words = set(stop_words)
				stop_words = list(stop_words)
				"""
				----
				"""
				vectorizer = CountVectorizer(stop_words=stop_words, 
										   analyzer='word', 
										   ngram_range=(1, 1),
										   max_features=n_feats)

			elif feat_type == 'tfidf_fwords': 
				"""
				Low-frequency function words gain higher weight
				Filters out words that are not function words
				"""
				stop_words = [t.split() for t in train_texts]
				stop_words = sum(stop_words, [])
				stop_words = [w for w in stop_words if w not in function_words_only]
				stop_words = set(stop_words)
				stop_words = list(stop_words)
				"""
				----
				"""
				vectorizer = TfidfVectorizer(stop_words=stop_words, 
										   analyzer='word', 
										   ngram_range=(1, 1),
										   max_features=n_feats)

			elif feat_type == 'raw_4grams': 
				vectorizer = CountVectorizer(stop_words=invalid_words, 
										   analyzer='char', 
										   ngram_range=(4, 4),
										   max_features=n_feats)

			elif feat_type == 'tfidf_4grams': 
				vectorizer = TfidfVectorizer(stop_words=invalid_words, 
										   analyzer='char', 
										   ngram_range=(4, 4),
										   max_features=n_feats)
			
			"""
			ENCODING X_TRAIN, x_test AND Y_TRAIN, y_test
			--------------------------------------------
	 		"""
			# Arranging dictionary where title is mapped to encoded label
			# Ultimately yields Y_train

			label_dict = {}
			inverse_label = {}
			for title in train_authors: 
				label_dict[title.split('_')[0]] = 0 
			for i, key in zip(range(len(label_dict)), label_dict.keys()):
				label_dict[key] = i
				inverse_label[i] = key

			"""
			TRAINING

			Step 1: input string is vectorized
				e.g. '... et quam fulgentes estis in summo sole ...'
			Step 2: to_dense = make sparse into dense matrix
			Step 3: feature scaling = normalize frequencies to chosen standard
			Step 4: reduce dimensionality by performing feature selection
			Step 5: choose type of classifier with specific decision function

			"""
			# Map Y_train to label_dict

			Y_train = []
			for title in train_authors:
				label = label_dict[title.split('_')[0]]
				Y_train.append(label)

			Y_train = np.array(Y_train)
			X_train = train_texts

			# DECLARING GRID, TRAINING
			# ------------------------
			"""
			Put this block of code in comment when skipping training and loading model
			Explicit mention labels=Y_train in order for next arg average to work
			average='macro' denotes that precision-recall scoring (in principle always binary) 
			needs to be averaged for multi-class problem
			"""

			pipe = Pipeline(
				[('vectorizer', vectorizer),
				 ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
				 ('feature_scaling', StandardScaler()),
				 ('reduce_dim', SelectKBest(mutual_info_regression)),
				 ('classifier', svm.SVC())])

			# c_options = [c_parameter]
			# n_features_options = [n_feats]
			kernel_options = ['linear']

			param_grid = [	
				{
					'vectorizer': [vectorizer],
					'feature_scaling': [StandardScaler()],
					'reduce_dim': [SelectKBest(mutual_info_regression)],
					'reduce_dim__k': n_selected_feats,
					'classifier__C': c_options,
					'classifier__kernel': kernel_options,
				},
			]

			# Change this parameter according to preferred high scoring metric
			refit = 'accuracy_score'

			for cv in cv_types:
				cv_name = str(cv).split('(')[0]

				print(":::{} as feature type, {} as number of features, {} sampled, {} cross-validation method ::::".format(feat_type, str(n_feats), sampling_type, cv_name))
				grid = GridSearchCV(pipe, cv=cv, n_jobs=9, param_grid=param_grid,
									scoring={
								 		'precision_score': make_scorer(precision_score, labels=Y_train, \
								 														 average='macro'),
										'recall_score': make_scorer(recall_score, labels=Y_train, \
										 										  average='macro'),
										'f1_score': make_scorer(f1_score, labels=Y_train, \
										 								  average='macro'),
										'accuracy_score': make_scorer(accuracy_score),},
									refit=refit, 
									# Refit determines which scoring method weighs through in model selection
									verbose=True
									# Verbosity level: amount of info during training
									) 

				# Get best model & parameters
				# Save model locally
				grid.fit(X_train, Y_train)
				model = grid.best_estimator_

				# Safety buffer: to avoid errors in code
				vectorizer = model.named_steps['vectorizer']
				classifier = model.named_steps['classifier']
				features = vectorizer.get_feature_names()
				best_c_param = classifier.get_params()['C']
				features_booleans = grid.best_params_['reduce_dim'].get_support()
				grid_features = list(compress(features, features_booleans))

				if len(features) != n_feats:
					sys.exit("ERROR: Inconsistent number of features: {} against {}".format(str(n_feats),str(len(features))))

				model_location = '/Users/jedgusse/compstyl/output/numpy_output/heloise-abelard-models/try_-{}-{}-{}-{}-{}-model'.format(feat_type, str(n_feats), sampling_type, cv_name, str(best_c_param))
				pickle.dump(grid, open(model_location, 'wb'))

				accuracy = grid.cv_results_['mean_test_accuracy_score'][0]
				precision = grid.cv_results_['mean_test_precision_score'][0]
				recall = grid.cv_results_['mean_test_recall_score'][0]
				f1 = grid.cv_results_['mean_test_f1_score'][0]

				results_file.write(str(accuracy) + '\t')
				results_file.write('\n')

"""
=======
TESTING
=======
"""

# train_authors, train_titles, train_texts = DataReader(train_location, sample_size,
# 												test_dict, rnd_dct
# 												).metadata(sampling=True,
# 												type='folder',
# 												randomization=False)
# label_dict = {}
# inverse_label = {}
# for title in train_authors: 
# 	label_dict[title.split('_')[0]] = 0 
# for i, key in zip(range(len(label_dict)), label_dict.keys()):
# 	label_dict[key] = i
# 	inverse_label[i] = key						

# _scores = []
# _performances = []
# _model_names = []
# _models = []
# _predictions = []

# for model in glob.glob('/Users/jedgusse/compstyl/output/numpy_output/heloise-abelard-models/*'):
# 	# Load model
# 	model_name = model.split('/')[-1]
# 	grid = pickle.load(open(model, 'rb'))
# 	best_model = grid.best_estimator_

# 	# Scores
# 	accuracy = grid.cv_results_['mean_test_accuracy_score'][0]
# 	precision = grid.cv_results_['mean_test_precision_score'][0]
# 	recall = grid.cv_results_['mean_test_recall_score'][0]
# 	f1 = grid.cv_results_['mean_test_f1_score'][0]
# 	average = np.mean([accuracy, precision, recall, f1])

# 	# Collect all model scores in containers
# 	_performances.append(average)
# 	_model_names.append(model_name)
# 	_scores.append([accuracy, precision, recall, f1])
# 	_models.append(best_model)

# 	"""
# 	DISCRETE SAMPLING
# 	Put in comment if shingling (cfr. below).
# 	"""

# 	# Make predictions
# 	test_location = '/Users/jedgusse/compstyl/impostors-method/abelard-heloise/test_set'

# 	test_dict = {}
# 	rnd_dct = {'n_samples': 140,
# 			   'smooth_train': True,
# 			   'smooth_test': False}
# 	test_authors, test_titles, test_texts = DataReader(test_location, sample_size,
# 										test_dict, rnd_dct
# 										).metadata(sampling=True,
# 										type='folder',
# 										randomization=False)

# 	"""
# 	ROLLING SAMPLING
# 	Put in comment if discrete (cfr. above)
# 	"""
# 	# dir_location = 'hildegard-vita/test_set'
# 	# test_data = load_and_split(dir_location)
# 	# test_set_location = '/Users/jedgusse/compstyl/impostors-method/hildegard-vita/test_set/Vita-Hildegardis_Vita-Hildegardis.txt'
# 	# original_test_txt = open(test_set_location).read()
# 	# test_txt = re.sub('[%s]' % re.escape(punctuation), '', original_test_txt) # Escape punctuation and make characters lowercase
# 	# text_txt = re.sub('\d+', '', test_txt)
# 	# test_txt = test_txt.lower().split()

# 	# # Collect test samples in dictionary by keyword of range
# 	# test_data = {}
# 	# # "shingling": make windows	
# 	# steps = np.arange(0, len(test_txt), step_size)

# 	# step_ranges = []
# 	# test_data = {}
# 	# key = dir_location.split('/')[-1].split('.')[0].split('_')[0]
# 	# test_data[key] = []
# 	# for each_begin in steps:
# 	# 	sample_range = range(each_begin, each_begin + sample_size)
# 	# 	step_ranges.append(sample_range)
# 	# 	text_sample = []
# 	# 	for index, word in enumerate(test_txt):
# 	# 		if index in sample_range:
# 	# 			text_sample.append(word)
# 	# 	title = '{}-{}'.format(str(each_begin), str(each_begin + sample_size))
# 	# 	if len(text_sample) == sample_size:
# 	# 		value = (title, ' '.join(text_sample))
# 	# 		test_data[key].append(value)

# 	# for key in test_data.keys():
# 	# 	test_titles = [i[0] for i in test_data[key]]
# 	# 	test_texts = [i[1] for i in test_data[key]]

# 	"""
# 	VECTORIZER
# 	----------
# 	"""
# 	# Call trained model's parameters and fit on test data
# 	vectorizer = best_model.named_steps['vectorizer']
# 	scaler = best_model.named_steps['feature_scaling']
# 	# dim_red = best_model.named_steps['reduce_dim']
# 	best_clf = best_model.named_steps['classifier']

# 	x_test = vectorizer.transform(test_texts).toarray()
# 	x_test = scaler.transform(x_test)
# 	# x_test = dim_red.transform(x_test)
	
# 	# Make prediction with this model
# 	y_pred = best_clf.predict(x_test)
# 	_predictions.append(list(y_pred))

# # Training results
# # Collect all models' results and order by highest performance (calculated through average precision, recall, accuracy and f1)
# results = []
# for model_name, performance, y_preds in zip(_model_names, _performances, _predictions):
# 	tup = (model_name, y_preds, performance)
# 	results.append(tup)
# sorted_results = sorted(results, key=lambda x: x[2])

# # Models ranked by performance
# # itemgetter gets first and last element in tuple, here: model name and performance of the model
# ranked_by_performance = [itemgetter(0,-1)(i) for i in sorted_results]
# df_1 = pd.DataFrame(ranked_by_performance, columns=['model name', 'average performance'])

# preds_per_sample = {}
# for title in test_titles:
# 	preds_per_sample[title] = []

# # Loop over prediction output of each of the models
# for (model_name, y_preds, performance) in sorted_results:
# 	y_authors = []
# 	# Loop over each of the models' individual predictions
# 	for title, pred_label in zip(test_titles, y_preds):
# 		pred_author = inverse_label[pred_label]
# 		preds_per_sample[title].append(pred_author)

# data_results = []
# for sample, pred_per_model in preds_per_sample.items():
# 	labels_n_freqs = Counter(pred_per_model).most_common(len(label_dict))
# 	print(sample, pred_per_model)
# 	winner = labels_n_freqs[0][0]
# 	number = labels_n_freqs[0][1]
# 	percentage = number / len(pred_per_model)
# 	data_tup = (sample, winner, percentage) 
# 	data_results.append(data_tup)

# # Consensus: scores of all models
# df_2 = pd.DataFrame(data_results, columns=['sample', 'prediction', 'confidence over all SVM models'])
# df_2.to_excel(excel_writer='/Users/jedgusse/compstyl/output/text_output/heloise-abelard-predictions.xlsx')

# print(df_1)

"""
================
PLOTTING COMET PLOT
================
"""

# labels = {0: 'Gualterus-de-Castellione', 
# 		  1: 'Guillelmus-de-Conchis',
# 		  2: 'Alanus-de-Insulis', 
# 		  3: 'Odo'}

# color_dict = {0: '#00A2FF', 
# 		  1: '#F9BA00',
# 		  2: '#60D836', 
# 		  3: 'r'}

# for label in list(range(0,4)):
	
# 	fig = plt.figure(figsize=(3,2))
# 	ax = fig.add_subplot(111)

# 	class_alphas = []
# 	for rank, (accuracy, y_pred) in enumerate(zip(sorted_performances, sorted_predictions)):
# 		# Number of wins per label expressed in percentage
# 		n_wins = [i for i in y_pred if label==i]
# 		alpha = len(n_wins) / len(y_pred)
# 		# Normalize alpha by confidence of model rank (better models gain higher weight)
# 		normalizing_constant = rank / len(sorted_performances)
# 		alpha = alpha * normalizing_constant
# 		class_alphas.append(alpha)
		
# 		ax.scatter(rank+1, accuracy, marker='s', alpha=alpha, color='k', s=0.3, linewidth=0.5)

# 	class_dominance = np.mean(class_alphas)

# 	rcParams['font.family'] = 'sans-serif'
# 	rcParams['font.sans-serif'] = ['Alfios']

# 	ax.set_ylabel('μ performance score')
# 	ax.set_xlabel('SVM rank (sorted by μ performance)')

# 	# ax.set_xticklabels(labels=['linear', 'polynomial', 'sigmoid', 'radial'])
# 	# ax.set_xlim(0, len(_model_names))
	
# 	# ax.set_yticks([0.85, 0.90, 0.95, 1.0])
# 	# ax.set_ylim(0.82, 1.0)

# 	for tick in ax.xaxis.get_major_ticks():
# 		tick.label.set_fontsize(7)
# 	for tick in ax.yaxis.get_major_ticks():
# 		tick.label.set_fontsize(7)

# 	# Despine
# 	ax.spines['right'].set_visible(False)
# 	ax.spines['top'].set_visible(False)
# 	ax.spines['left'].set_visible(True)
# 	ax.spines['bottom'].set_visible(True)

# 	print(class_dominance)

# 	plt.tight_layout()
# 	plt.show()

# 	fig.savefig("/Users/jedgusse/compstyl/output/fig_output/{}.pdf".format(labels[label]), \
# 				transparent=True, format='pdf')

"""
===============================
PLOTTING ACCURACY & LOWESS LINE
================================
"""

# s_ls = list(range(50,5050,50))
# # feat_loop = [75, 150, 400, 1000]

# x = list(range(2,len(train_texts)+1))
# # x = s_ls

# settings = ['raw_MFW', 'tfidf_MFW', 'raw_4grams', 'tfidf_4grams', 'raw_fwords', 'tfidf_fwords']

# legend_dictionary = {'raw_MFW': 'MFW',
# 				  'tfidf_MFW': 'tfidf MFW',
# 				  'raw_fwords': 'function words',
# 				  'tfidf_fwords': 'tfidf function words',
# 				  'raw_4grams': '4-grams',
# 				  'tfidf_4grams': 'tfidf 4-grams'}

# customized_colors = {'raw_MFW': '#EF5FA7',
# 				  'tfidf_MFW': 'k',
# 				  'raw_fwords': '#EE220C',
# 				  'tfidf_fwords': '#61D836',
# 				  'raw_4grams': '#00A2FF',
# 				  'tfidf_4grams': '#F8BA00'}

# # y0 = [(0.7690040335091529, 0.7699363560745631, 0.7690081308451036, 0.7684722625470135), (0.8073936004970488, 0.8092791969666638, 0.8073964873729221, 0.8066649605785842), (0.8439468159552135, 0.8457547085418297, 0.8439474226593331, 0.843415211563664), (0.8714196762141968, 0.8742297354411681, 0.8714001753257358, 0.870741605844459), (0.9058365758754864, 0.9079290658553931, 0.9057973294270211, 0.9052266716817379), (0.9063670411985019, 0.9088886512608282, 0.9063232585798727, 0.906018731226165), (0.9092400218698743, 0.9115190553466627, 0.9092139086756936, 0.9086071690196401), (0.9237976264834479, 0.9283931612339807, 0.9237899331946421, 0.9233291180895512), (0.9408450704225352, 0.9459839776136226, 0.940882179805182, 0.9402973039218209), (0.9351055512118843, 0.9391449157822699, 0.9350548508510681, 0.9347425057545107), (0.9396031061259706, 0.9429398490279818, 0.9396013773110212, 0.9390487976340411), (0.9464285714285714, 0.9498577717447777, 0.9464818793098234, 0.9460034979242835), (0.9459183673469388, 0.9508678232691982, 0.9458372350191819, 0.9449920894161319), (0.9448123620309051, 0.9513021728456367, 0.9448071014708176, 0.9440416907108303), (0.9456906729634003, 0.9512172734147915, 0.9457103440881721, 0.9454301427781111), (0.9508196721311475, 0.9539755528490433, 0.9507813083608901, 0.9506987983166464), (0.9491298527443106, 0.9532260112861533, 0.9491603270048514, 0.9487609177635867), (0.957325746799431, 0.961640226476799, 0.9572510629761238, 0.9568311436705705), (0.9416167664670658, 0.9478643052325051, 0.9416214004581722, 0.941104292519843), (0.95260663507109, 0.9572606551229422, 0.9526315920826377, 0.9522284754697251), (0.95, 0.9561767225190283, 0.9499003026947213, 0.9490126957965184), (0.951048951048951, 0.9560031862115146, 0.9508982949021122, 0.950258930168744), (0.948905109489051, 0.9536848458891282, 0.948828224436275, 0.948287613380036), (0.9581749049429658, 0.9634658901043217, 0.9580426430739488, 0.9576799908091754), (0.9581673306772909, 0.9621245358165273, 0.9580988230784924, 0.9577061222673693), (0.9607438016528925, 0.9630031554363712, 0.9609694431768863, 0.9606424909637061), (0.9547413793103449, 0.9587544020047477, 0.9547150997011704, 0.954381465271827), (0.953125, 0.9579562835918265, 0.952918770283441, 0.9524798948795763), (0.9605568445475638, 0.9656556557686156, 0.9605354851560044, 0.9599161097269946), (0.9736842105263158, 0.9757296044916822, 0.9735903535686369, 0.9734948346829492), (0.9653465346534653, 0.9677983460656485, 0.96522085693913, 0.9651547570320846), (0.9692307692307692, 0.9717213796214587, 0.9690491996528321, 0.9689943660504323), (0.9706666666666667, 0.9748945663082438, 0.9704148996109426, 0.9702843305379497), (0.9700272479564033, 0.9722323446923203, 0.9701283490433597, 0.9698604973225943), (0.9661971830985916, 0.9689356686931864, 0.9663905765823693, 0.9659217698218877), (0.9710144927536232, 0.9730594899769308, 0.971000481263854, 0.9707849813142818), (0.9613095238095238, 0.9652170698010607, 0.9612730616596527, 0.960909895770156), (0.9603658536585366, 0.9657202301929424, 0.9603751038173161, 0.9598358544379619), (0.9561128526645768, 0.9597944834324684, 0.9560534866278976, 0.9554057910688106), (0.9615384615384616, 0.9653158585888266, 0.9612813068341916, 0.9610146436646567), (0.9668874172185431, 0.9703127691450095, 0.9668537549902402, 0.9665020455173958), (0.9591836734693877, 0.9651770765990647, 0.9594028281866681, 0.9584088657266375), (0.96875, 0.9731554144802599, 0.9687378001451352, 0.9682889597312871), (0.9644128113879004, 0.9704387889134422, 0.9640930499333782, 0.9634942627002288), (0.9635036496350365, 0.9663577629059275, 0.9635695693837539, 0.9631834532741413), (0.9740740740740741, 0.9777638456491918, 0.974399117313771, 0.974008905900159), (0.9618320610687023, 0.9675944357489721, 0.9619186902128793, 0.9612557831619937), (0.95703125, 0.9609648557133113, 0.9566582321220096, 0.9558724445053636), (0.9717741935483871, 0.9763983344506917, 0.971749599944036, 0.9716163723784266), (0.9795081967213115, 0.9819281313246544, 0.9794515824335648, 0.9793015784663244), (0.975103734439834, 0.9788317404278418, 0.9752070384463079, 0.9748685132491233), (0.961864406779661, 0.96365635793016, 0.9618876518860011, 0.9618068110815509), (0.9655172413793104, 0.968603622665606, 0.9651353098609665, 0.9652502198697992), (0.9736842105263158, 0.9767170007243876, 0.973741920590951, 0.9735451678086346), (0.9732142857142857, 0.9755304651622554, 0.9730548731622584, 0.9729900568349229), (0.9678899082568807, 0.9715677545011422, 0.9679263810565891, 0.9677899995690872), (0.9720930232558139, 0.9768451415179377, 0.9716387236343971, 0.9717724144973469), (0.9714285714285714, 0.9750285974038739, 0.971285930245114, 0.9711117454157806), (0.9803921568627451, 0.9828513205804388, 0.980021420332839, 0.9802347089958826), (0.9754901960784313, 0.9792893421558511, 0.9749752842313396, 0.9752301924186807), (0.9702970297029703, 0.972870790531372, 0.970261936751958, 0.9700036419209744), (0.9693877551020408, 0.9725270720533111, 0.9689669354229694, 0.969025469617015), (0.9791666666666666, 0.9816514756944446, 0.9790962414863782, 0.9789930099644023), (0.9735449735449735, 0.9764792846395859, 0.973891408415218, 0.9731946806352066), (0.9732620320855615, 0.9773170522462754, 0.9730547055963854, 0.9727278793251511), (0.9723756906077348, 0.9767009554042917, 0.9722764872867129, 0.97196812121265), (0.9776536312849162, 0.9809450024509039, 0.9773269664908503, 0.9775793555091955), (0.9662921348314607, 0.9713250378740058, 0.9659461703553358, 0.9658440156875922), (0.9655172413793104, 0.9705231478243298, 0.965208966398027, 0.9647947847443249), (0.9767441860465116, 0.9802608233874737, 0.9762934919776455, 0.975958168835671), (0.9707602339181286, 0.9746602717686581, 0.9705379432987928, 0.9700652087393167), (0.9640718562874252, 0.9703244390456089, 0.963899248706879, 0.962736239539164), (0.9636363636363636, 0.969354650400105, 0.9640568614122332, 0.9634537467439198), (0.968944099378882, 0.9722422864739905, 0.9693347301850059, 0.9686241441374369), (0.9493670886075949, 0.9549724038512895, 0.9494758166044819, 0.9486957204416426), (0.9681528662420382, 0.97236211798263, 0.9678572877485611, 0.9678042505715448), (0.9545454545454546, 0.9588709450722437, 0.9542370262559164, 0.9541515106119446), (0.9539473684210527, 0.9586704359957909, 0.9541047595064216, 0.9532961007075058), (0.9727891156462585, 0.9738036501884899, 0.9729101478372616, 0.9727624510929714), (0.9659863945578231, 0.9684212774591783, 0.9664313658482755, 0.965959560967489), (0.9659863945578231, 0.9695328966465993, 0.9664313658482756, 0.9659321962368208), (0.9655172413793104, 0.9685100155492546, 0.9652282081770787, 0.9652566723525481), (0.9583333333333334, 0.9621343644781146, 0.9576613940329219, 0.9579980402701823), (0.958041958041958, 0.9618607711423987, 0.957649110795964, 0.9578533995576204), (0.9716312056737588, 0.9764599366229062, 0.9717066545948393, 0.9715344176638879), (0.9640287769784173, 0.9689656133421983, 0.9636820040370581, 0.9633752186574721), (0.9635036496350365, 0.9688110919396551, 0.9633864350791198, 0.9631292020683263), (0.9701492537313433, 0.9738023170858991, 0.9701405905794412, 0.9698478988132528), (0.9699248120300752, 0.9731488930539188, 0.9704433389237505, 0.9696471853572785), (0.9699248120300752, 0.9731488930539188, 0.9704433389237505, 0.9696471853572785), (0.9699248120300752, 0.9731488930539188, 0.9704433389237505, 0.9696471853572785), (0.9769230769230769, 0.9800591715976331, 0.9769230769230769, 0.9766218430692115), (0.96875, 0.9725341796875, 0.9696266867897727, 0.9688036328298109), (0.968, 0.9721088, 0.9680570505050505, 0.967571056802667), (0.967479674796748, 0.971056323029355, 0.9684763224818747, 0.9672184170024483), (0.9669421487603306, 0.970911215839841, 0.9677360656631198, 0.9667284386966757), (0.9663865546218487, 0.972636113268837, 0.966471294400113, 0.9660524255538733), (0.9568965517241379, 0.9643561056462364, 0.9552509767283845, 0.9554194269860143), (0.9568965517241379, 0.9643561056462364, 0.9552509767283845, 0.9554194269860143), (0.9478260869565217, 0.9583093315002578, 0.9468279773156899, 0.9452982757633042)]
# # y1 = [(0.7556624263108904, 0.756734610952754, 0.755665510828915, 0.7554275852018013), (0.8039763901832867, 0.8058277583322653, 0.8039794072263959, 0.8035949581639104), (0.8252857476090506, 0.8282450445924526, 0.825285032268037, 0.8249363409066771), (0.8676836861768369, 0.8705163192529198, 0.8676667786944722, 0.8672867862221866), (0.8926070038910506, 0.895735572895948, 0.8925626379403954, 0.8920947206246671), (0.899812734082397, 0.9033202345451175, 0.8997691980188387, 0.8995727815663339), (0.9015855658829962, 0.9052030079923697, 0.9015465586412899, 0.901035767589517), (0.9231730168644597, 0.9268032556432246, 0.9231599173040033, 0.9229735260782371), (0.9345070422535211, 0.9393596080557961, 0.9345478105497612, 0.9340792324389784), (0.9202501954652071, 0.9238814475829689, 0.9202080633311134, 0.9200952055851792), (0.9439171699741156, 0.9473568028316461, 0.9439157076366356, 0.943645972058237), (0.9426691729323309, 0.9462800051812721, 0.9427210506470546, 0.9424234829496269), (0.95, 0.955331721145464, 0.9499138999084232, 0.949580491803928), (0.9437086092715232, 0.9475921267080276, 0.9436962040454493, 0.9434623004842467), (0.9456906729634003, 0.9510688993130002, 0.9457148748495474, 0.9454977392554048), (0.9482976040353089, 0.9524179423607663, 0.9482815349464775, 0.948176072685058), (0.9544846050870147, 0.957144849761528, 0.9545058884896813, 0.9542998988235101), (0.9601706970128022, 0.9636294395469228, 0.9601394492961003, 0.9602071864016994), (0.9491017964071856, 0.9555289675088925, 0.949102189347117, 0.9488285159717327), (0.95260663507109, 0.956143951668064, 0.9525816780595424, 0.9523652617004785), (0.945, 0.9523933862433862, 0.9449031360280549, 0.9438345280858499), (0.9562937062937062, 0.9614775343192118, 0.9561108589298978, 0.9558945195426718), (0.9562043795620438, 0.9611379592378104, 0.9561207605922296, 0.9557808815652993), (0.9562737642585551, 0.9616856653846299, 0.9561246029270778, 0.9557600232145554), (0.9581673306772909, 0.9621245358165273, 0.9580988230784924, 0.9577061222673693), (0.9607438016528925, 0.9635361945866439, 0.9609737371217709, 0.9605996104138423), (0.9655172413793104, 0.9688377167131358, 0.9654563704706176, 0.9653801939632507), (0.9553571428571429, 0.9608510703825789, 0.9551328517167956, 0.954690933725433), (0.962877030162413, 0.967105934623279, 0.9628431098278498, 0.9626128534226267), (0.9760765550239234, 0.9773486595760583, 0.9760222255721446, 0.975984059423441), (0.9653465346534653, 0.9685646550162039, 0.9651904339582863, 0.9651364568442383), (0.9666666666666667, 0.9694353840383617, 0.9664852362341678, 0.9663706537880427), (0.968, 0.9723292329749105, 0.9677433098673529, 0.9676193171548187), (0.9700272479564033, 0.9725493914951283, 0.9701815181241503, 0.9698822165982823), (0.9577464788732394, 0.9620679084397118, 0.9579922429184138, 0.957287894121822), (0.9681159420289855, 0.9718808373590982, 0.9681832927566582, 0.9679817499458687), (0.9583333333333334, 0.9646650815716394, 0.9582604795875821, 0.9577966068124145), (0.9664634146341463, 0.9704091907807145, 0.9664632350194127, 0.9661360186985722), (0.9623824451410659, 0.9662619383259871, 0.9623773525639919, 0.9620984547745969), (0.9647435897435898, 0.9679333561118246, 0.964550518036095, 0.9645426608003967), (0.9635761589403974, 0.9676223663768071, 0.9635381109395407, 0.9631826717220165), (0.9591836734693877, 0.9675876237465161, 0.9594772019595009, 0.9580254820515232), (0.9652777777777778, 0.9705965027227896, 0.9652414652685919, 0.9647615791750792), (0.9715302491103203, 0.9761255477879204, 0.9712957062747166, 0.9709454030805109), (0.9708029197080292, 0.9756591903135579, 0.9708882946745162, 0.9704320675154836), (0.9703703703703703, 0.9750078680191402, 0.9707640007156914, 0.9702152905057772), (0.9580152671755725, 0.9639535169481864, 0.9580188297705681, 0.9576579661725961), (0.96484375, 0.9711059275456673, 0.9643933482631926, 0.9635814548833429), (0.9637096774193549, 0.9708063333648661, 0.963761597250763, 0.9631571816181975), (0.9590163934426229, 0.9663544106167058, 0.9589031648671293, 0.9582237057849882), (0.975103734439834, 0.9786834428380755, 0.9752070384463079, 0.9750396441072356), (0.9661016949152542, 0.9712738197835871, 0.9660274473190598, 0.9657050641620187), (0.9655172413793104, 0.9711733223177932, 0.9649866197603564, 0.9651261612606035), (0.9736842105263158, 0.9776083568591586, 0.973741920590951, 0.9735453472000571), (0.9732142857142857, 0.9773012529729936, 0.9731423284774436, 0.9729755120340261), (0.9770642201834863, 0.9803053895575569, 0.9771084083831327, 0.9770180514757215), (0.9720930232558139, 0.9768451415179377, 0.9716387236343971, 0.9717724144973469), (0.9666666666666667, 0.9715394239567173, 0.9665731562466257, 0.9663677948052332), (0.9705882352941176, 0.975634371395617, 0.9700321304992585, 0.9701636717235556), (0.9656862745098039, 0.9692223638681479, 0.9652262865930686, 0.9653679435137559), (0.9702970297029703, 0.9751224080198536, 0.9701771699759211, 0.9699372347692404), (0.9591836734693877, 0.9635384902867852, 0.9585673902185904, 0.9587283407046006), (0.9791666666666666, 0.9816514756944446, 0.9790962414863782, 0.9789930099644023), (0.9735449735449735, 0.9762099627230865, 0.9738179222306207, 0.9734207953753179), (0.9679144385026738, 0.9736087677657354, 0.9676892390402929, 0.9673781820727736), (0.9779005524861878, 0.9807334282572954, 0.9777962618153699, 0.9777460209326606), (0.9664804469273743, 0.9731021711765134, 0.9659904497362755, 0.966083619938756), (0.9719101123595506, 0.977044453686032, 0.9715418928586458, 0.9718396736839772), (0.9770114942528736, 0.9801740652662174, 0.9769013960452725, 0.9768349136660671), (0.9767441860465116, 0.9798497498647918, 0.9765188390120785, 0.9764519282772393), (0.9707602339181286, 0.9743232960569064, 0.9706633380071362, 0.9705436178462511), (0.9700598802395209, 0.9750541688223213, 0.9698330999560875, 0.9693922237858567), (0.9636363636363636, 0.9695426997245179, 0.9638021299178324, 0.9635813969179651), (0.9627329192546584, 0.9679333359052505, 0.9631511063168792, 0.9622808466032131), (0.9556962025316456, 0.9601388435244187, 0.9560853343099778, 0.9553545110705917), (0.9617834394904459, 0.9679451030935878, 0.9612850130112492, 0.9608976028839101), (0.9675324675324676, 0.9703505369089785, 0.9673927017492543, 0.9675095811128721), (0.9671052631578947, 0.9684669771018456, 0.9673492193402167, 0.9671021775738206), (0.9591836734693877, 0.9621811618896167, 0.9597033999366361, 0.959216949143661), (0.9591836734693877, 0.9621811618896167, 0.9597033999366361, 0.959216949143661), (0.9659863945578231, 0.9695328966465993, 0.9664313658482756, 0.9659321962368208), (0.9586206896551724, 0.9622224624364932, 0.9584231226561785, 0.9584493916427704), (0.9583333333333334, 0.9621343644781146, 0.9576613940329219, 0.9579980402701823), (0.958041958041958, 0.9618607711423987, 0.957649110795964, 0.9578533995576204), (0.9716312056737588, 0.9764599366229062, 0.9717066545948393, 0.9715344176638879), (0.9640287769784173, 0.9689656133421983, 0.9636820040370581, 0.9633752186574721), (0.9635036496350365, 0.9688110919396551, 0.9633864350791198, 0.9631292020683263), (0.9626865671641791, 0.9682500242758652, 0.9627335957030767, 0.9624593210836482), (0.9548872180451128, 0.9604224426147656, 0.9558438703274478, 0.9545433637791233), (0.9624060150375939, 0.9661671529894719, 0.9630941388559118, 0.9619958837846747), (0.9624060150375939, 0.9661671529894719, 0.9630941388559118, 0.9619958837846747), (0.9692307692307692, 0.9730078895463511, 0.9692307692307692, 0.9687471091075212), (0.96875, 0.9725341796875, 0.9696266867897727, 0.9688036328298109), (0.96, 0.9662545454545455, 0.9598883232323233, 0.9594750086823662), (0.959349593495935, 0.9650340405843082, 0.9603642679688683, 0.9590018093643945), (0.9586776859504132, 0.9646882043576259, 0.9596640884633508, 0.9584168473206038), (0.9663865546218487, 0.972636113268837, 0.966471294400113, 0.9660524255538733), (0.9568965517241379, 0.9643561056462364, 0.9552509767283845, 0.9554194269860143), (0.9568965517241379, 0.9643561056462364, 0.9552509767283845, 0.9554194269860143), (0.9565217391304348, 0.9636143667296786, 0.9556748582230623, 0.9552010081915564)]
# # y2 = [(0.7145516599441514, 0.7154090887572634, 0.7145539529676176, 0.7136011367603142), (0.7556694625660143, 0.7564726778314387, 0.7556723953943681, 0.7551082321427336), (0.7863307674364357, 0.7878786976984522, 0.7863286599851707, 0.7856200734417608), (0.8259651307596513, 0.8277278430212244, 0.825964795254654, 0.8254907218204889), (0.8396887159533074, 0.8411116799759041, 0.8396754454294816, 0.8390705459608667), (0.8661048689138576, 0.8668296091403094, 0.8660504719910134, 0.8654215816335321), (0.8840896664844177, 0.8853929457326989, 0.8840880610830079, 0.8833998839043584), (0.8988132417239225, 0.9004768628988544, 0.8987865907418409, 0.8978359495040101), (0.9035211267605634, 0.9058317693926986, 0.903575552048318, 0.9027880939483582), (0.9171227521501173, 0.9189503638214755, 0.9170910328443399, 0.9167039176916829), (0.913718723037101, 0.9156643160238463, 0.9136566678667855, 0.9130756197623953), (0.918233082706767, 0.9208574909820935, 0.918237689132192, 0.917810067686352), (0.926530612244898, 0.9294015063007038, 0.9264250697802042, 0.9257997456961019), (0.9337748344370861, 0.9354948171926223, 0.9337314353165364, 0.9333560876420293), (0.9362455726092089, 0.9387711116685175, 0.9363678104728017, 0.9359979699488445), (0.9419924337957125, 0.9441775551207005, 0.9419624633711245, 0.9418971171741034), (0.9464524765729585, 0.9486381664279636, 0.9464872402463894, 0.9463887702921144), (0.9459459459459459, 0.9468918115004566, 0.9459079427208744, 0.9456803575757694), (0.9431137724550899, 0.9453480936933213, 0.9430844825015894, 0.942839338098153), (0.9557661927330173, 0.9569554916476972, 0.9557973058074134, 0.9555251363043922), (0.9416666666666667, 0.942849836900329, 0.9417213069398301, 0.9415676048187213), (0.9527972027972028, 0.9553496733477375, 0.9527405768655266, 0.9525010280919999), (0.9708029197080292, 0.972203215561741, 0.9706809100111887, 0.9707557764732071), (0.9619771863117871, 0.9646859905379522, 0.9619737772932165, 0.9618292402352873), (0.9601593625498008, 0.9617841930996424, 0.9600665484537038, 0.9600209128663758), (0.96900826446281, 0.9704944258220425, 0.9691322324029515, 0.9689955950126947), (0.9612068965517241, 0.9625896658731828, 0.961194551246949, 0.9611408571103731), (0.9553571428571429, 0.9564966655730205, 0.9554308149828733, 0.9553249011151334), (0.9651972157772621, 0.9660584588127564, 0.965167698801586, 0.9651186299187187), (0.9688995215311005, 0.9703123204178495, 0.9687844385095067, 0.9687222529879504), (0.9554455445544554, 0.9560817408554192, 0.9553661502524766, 0.9553664188146545), (0.9692307692307692, 0.9695312232745806, 0.9690926557219293, 0.9691292848418068), (0.9653333333333334, 0.9663668604748732, 0.9651812586141341, 0.965160694199589), (0.9754768392370572, 0.9761855287185905, 0.9755853648352952, 0.9754698976510282), (0.9690140845070423, 0.9700141066258086, 0.9690946554434012, 0.968916760497421), (0.9768115942028985, 0.9775123458897803, 0.976874216859824, 0.9768870581839699), (0.9672619047619048, 0.9687704570236145, 0.967353122921994, 0.9672115898836191), (0.9695121951219512, 0.9704149225971469, 0.9695221038680872, 0.9694879150426898), (0.9717868338557993, 0.9723755148033605, 0.9718396339870442, 0.9717774939751682), (0.9743589743589743, 0.9757985767370899, 0.9742310974282129, 0.9742136221993716), (0.9668874172185431, 0.9685399377605861, 0.9668940243564164, 0.9667414527556925), (0.9795918367346939, 0.98047818679847, 0.9796750114039785, 0.9796423397248736), (0.9791666666666666, 0.9798420401936025, 0.9790881570124191, 0.9790200540859834), (0.9750889679715302, 0.9758380719595751, 0.9750458087570214, 0.9750226934629375), (0.9817518248175182, 0.9827437064731344, 0.9816715067889379, 0.9817969779047221), (0.9666666666666667, 0.9671697969134898, 0.9666900350797292, 0.9666748057705067), (0.9732824427480916, 0.9738728130594759, 0.9733936963939817, 0.9733481042944369), (0.96875, 0.9711948399801342, 0.96870824459119, 0.968558813016762), (0.9637096774193549, 0.9673254783830839, 0.9638716764020905, 0.9636678730998434), (0.9631147540983607, 0.9667015822037117, 0.962931987464796, 0.9627821170267717), (0.970954356846473, 0.9726826924234538, 0.9709136613287713, 0.9708852626632517), (0.9830508474576272, 0.9844590559276676, 0.9830714952599828, 0.9830952122526223), (0.9827586206896551, 0.9851153760404281, 0.982426825344984, 0.9826416978266735), (0.9824561403508771, 0.9827623243049144, 0.9823984302862419, 0.9823401951507125), (0.9821428571428571, 0.9826440718831392, 0.9820817651690612, 0.9821118538454207), (0.9724770642201835, 0.973993690787617, 0.9725140046198881, 0.9724459133007549), (0.9767441860465116, 0.9797144022847817, 0.9765480439877411, 0.9766864884486711), (0.9761904761904762, 0.9775984692122462, 0.9761336788683729, 0.9761263316938104), (0.9803921568627451, 0.9828513205804388, 0.980021420332839, 0.9802347089958826), (0.9803921568627451, 0.9828513205804388, 0.980021420332839, 0.9802347089958826), (0.9801980198019802, 0.9818683673842049, 0.9800929057045816, 0.9800793605734763), (0.9744897959183674, 0.9756622546977974, 0.9742072828514676, 0.9744402018143377), (0.9739583333333334, 0.9762229370915034, 0.9740602076417506, 0.9739001260489467), (0.9682539682539683, 0.9712664215205972, 0.9684989222026259, 0.9680581814433503), (0.9786096256684492, 0.9807722133721801, 0.9785694802209429, 0.9785689907470375), (0.988950276243094, 0.9904113631045043, 0.9888358108726839, 0.9889089960333207), (0.9776536312849162, 0.9803345713304829, 0.9773724810919342, 0.9776385088219292), (0.9887640449438202, 0.9906072465597778, 0.9886377982577957, 0.9888812740094145), (0.9827586206896551, 0.9833963433844528, 0.9829100057251068, 0.9829656812497436), (0.9767441860465116, 0.9792020179826934, 0.9766850324499728, 0.9766991830512425), (0.9824561403508771, 0.9845901302965016, 0.9823991427561757, 0.9824607342835764), (0.9760479041916168, 0.9786357140295969, 0.9760406087273309, 0.9760139030273048), (0.9696969696969697, 0.9728983339892431, 0.9695765208161902, 0.9696547426209869), (0.9627329192546584, 0.9638944649090864, 0.9628534987510334, 0.9624102064078405), (0.9810126582278481, 0.9822775272582998, 0.9810469933847598, 0.9809038816137974), (0.9745222929936306, 0.9774101349986781, 0.974429562485873, 0.9744521399882452), (0.974025974025974, 0.9774570512974966, 0.9735718826627917, 0.973980027668464), (0.9671052631578947, 0.9684669771018456, 0.9673492193402167, 0.9671021775738206), (0.9795918367346939, 0.9811626662938617, 0.9798516999683181, 0.9796279258468258), (0.9591836734693877, 0.9601341321457939, 0.9592406297945658, 0.9590929330502752), (0.9795918367346939, 0.9800510471064408, 0.9798516999683181, 0.9796552905774941), (0.9724137931034482, 0.9737596529510919, 0.9722198847525839, 0.9723606296109295), (0.9722222222222222, 0.9736244569794918, 0.9718597286522636, 0.9721081020973745), (0.965034965034965, 0.9671712218536025, 0.9646910199357753, 0.9650213301935762), (0.9787234042553191, 0.9819006421541505, 0.9786814881880522, 0.9787637278823648), (0.9856115107913669, 0.9879923399409969, 0.9852405845108085, 0.9853997769737121), (0.9781021897810219, 0.9791144973093932, 0.9784183138863729, 0.978484411159011), (0.9626865671641791, 0.9648382915814416, 0.9630553699165861, 0.9628071440125229), (0.9924812030075187, 0.9929786873198033, 0.9927497314715359, 0.9925218234761212), (0.9849624060150376, 0.9859969472553564, 0.9854005314036972, 0.9848705219035175), (0.9849624060150376, 0.9859969472553564, 0.9854005314036972, 0.9848705219035175), (0.9769230769230769, 0.9775641025641025, 0.9769230769230769, 0.9767406506536941), (0.9765625, 0.9767913818359375, 0.9763405539772727, 0.9762206406849425), (0.976, 0.9760800000000001, 0.9759308282828282, 0.9756528286445014), (0.983739837398374, 0.9834191107986467, 0.9834191107986467, 0.9834191107986467), (0.9834710743801653, 0.983176912903366, 0.983176912903366, 0.983176912903366), (0.9831932773109243, 0.9849372904241911, 0.9832356472000565, 0.9831874784932315), (0.9741379310344828, 0.9746333729686881, 0.9734690844233056, 0.9736437162010434), (0.9655172413793104, 0.9672017439556084, 0.9643600305758452, 0.9648460829979405), (0.9826086956521739, 0.9858449905482042, 0.9822155009451796, 0.9822558286074354)]
# # y3 = [(0.7154049022649706, 0.7161446560974312, 0.7154060030839561, 0.714816813840024), (0.7545821683752718, 0.756163871440285, 0.7545835623568569, 0.7542215253340432), (0.7921623512946117, 0.794317240528408, 0.792159027554277, 0.7919499398616895), (0.8293897882938979, 0.8318755954267233, 0.8293978590153755, 0.8290817493520595), (0.843579766536965, 0.8454185044270915, 0.8435713073597237, 0.8429072317646231), (0.8656367041198502, 0.8671382274402251, 0.8655770772212189, 0.8651697634439288), (0.8906506287588847, 0.8917680850500215, 0.8906463989517678, 0.890154231022068), (0.8988132417239225, 0.9001486314217721, 0.8987904884827556, 0.8982229798453346), (0.9056338028169014, 0.9082743388598136, 0.905694984399635, 0.9051123587013226), (0.9186864738076622, 0.919912926686305, 0.9186688114082906, 0.9183363055251519), (0.913718723037101, 0.9150227618033676, 0.9136373782749796, 0.9130811265076731), (0.9229323308270677, 0.9245416575244713, 0.9229413857149976, 0.922728043539856), (0.9255102040816326, 0.9271370911161742, 0.9254308427029916, 0.9250937401124952), (0.9370860927152318, 0.9384057472852826, 0.9370499994924095, 0.9367613944355643), (0.9421487603305785, 0.9444132273251117, 0.9422710172365895, 0.9419561311390297), (0.9520807061790668, 0.9533938221667692, 0.952037574700252, 0.9519885428772428), (0.9531459170013387, 0.9546330084586969, 0.9531659971285126, 0.9530415405690462), (0.9530583214793741, 0.9543053782146835, 0.9530163217058987, 0.9527518872776914), (0.9520958083832335, 0.9535041771956506, 0.9520512731673632, 0.9520821478973241), (0.9620853080568721, 0.9632020227406463, 0.9621239082347325, 0.9618972950794313), (0.9466666666666667, 0.9478516688323201, 0.9466908482834995, 0.9465733379799192), (0.9615384615384616, 0.9631288053575016, 0.9614945401012507, 0.961416465741233), (0.9744525547445255, 0.9755930642496041, 0.9743438648835847, 0.9744202820874233), (0.9695817490494296, 0.9710781981554636, 0.9695863296154845, 0.9695491270401656), (0.9681274900398407, 0.9696526501085841, 0.9680497819074784, 0.9680164587471237), (0.9710743801652892, 0.9719054972740958, 0.971192265016844, 0.9710679340255568), (0.9698275862068966, 0.9714630959770089, 0.9698338199746542, 0.9697938195426479), (0.96875, 0.9696561817587834, 0.9687613310823322, 0.968788569643884), (0.9605568445475638, 0.9610434213294603, 0.9605396321691159, 0.9605063918803393), (0.9665071770334929, 0.9676328062685113, 0.9663723341141187, 0.9663440193437853), (0.9678217821782178, 0.9687052981672403, 0.9677636634329343, 0.9677988688023167), (0.9717948717948718, 0.9725700623522708, 0.971638747648363, 0.9717131266956116), (0.968, 0.9692552536199432, 0.9678042175333511, 0.9679332777011652), (0.9782016348773842, 0.9788628394200977, 0.978295311453059, 0.9782100883531923), (0.9690140845070423, 0.9696959048679868, 0.9690862574633363, 0.9689531779527939), (0.9797101449275363, 0.9808900987739395, 0.9797442021570305, 0.9797549013457104), (0.9732142857142857, 0.974425368414939, 0.9733047247239801, 0.9732355549586958), (0.975609756097561, 0.9769895280198111, 0.9756164317784923, 0.9755884101977005), (0.9780564263322884, 0.9789172472951132, 0.9781018465425194, 0.978061144090783), (0.9807692307692307, 0.9820738687258452, 0.9807577349644657, 0.9807639648130186), (0.9735099337748344, 0.9760415928676095, 0.9734950677146195, 0.9733860447998268), (0.9829931972789115, 0.9835188239498944, 0.983046622439063, 0.9830653576665911), (0.9826388888888888, 0.9840749547101448, 0.9826194552377276, 0.9825215066176975), (0.9822064056939501, 0.9835260445029825, 0.9821569142394747, 0.9820886252355903), (0.9854014598540146, 0.9865068219671786, 0.9853296180846429, 0.9854200092115601), (0.9777777777777777, 0.979624957754319, 0.9779358263255203, 0.9777729911465155), (0.9732824427480916, 0.9738728130594759, 0.9733936963939817, 0.9733481042944369), (0.9765625, 0.9782698318421135, 0.9765546530113288, 0.976404094046973), (0.9717741935483871, 0.9743499947633014, 0.9718735471555948, 0.9717996410992106), (0.9713114754098361, 0.9738173912891732, 0.9711033492640173, 0.9710553250282308), (0.979253112033195, 0.9806354611757955, 0.97926406851873, 0.9792603569402466), (0.9872881355932204, 0.9884647295846626, 0.9873559142487791, 0.9873607168980696), (0.9870689655172413, 0.9891284557074911, 0.986763877589336, 0.9869381659772564), (0.9868421052631579, 0.9870014051583494, 0.9867266851338873, 0.9867484268453798), (0.9866071428571429, 0.9867936758036547, 0.9865952111724624, 0.9865772725705761), (0.981651376146789, 0.9822274253217471, 0.9816827053465385, 0.981691261753108), (0.9813953488372092, 0.9842232406326796, 0.9812439156300704, 0.9814307369959072), (0.9809523809523809, 0.9816190365034904, 0.9809814274916316, 0.981015930295427), (0.9803921568627451, 0.9828513205804388, 0.980021420332839, 0.9802347089958826), (0.9803921568627451, 0.9828513205804388, 0.980021420332839, 0.9802347089958826), (0.9851485148514851, 0.9866146340900834, 0.9850335977935926, 0.9850758281273673), (0.9846938775510204, 0.9865307730234951, 0.9844708635023869, 0.9846899019347998), (0.9791666666666666, 0.9811057495915034, 0.9791755350227032, 0.9790636787385409), (0.9735449735449735, 0.9757790290327379, 0.9736779485456735, 0.9734503553490647), (0.983957219251337, 0.9858220642485025, 0.9838356830335441, 0.9839088545398167), (0.988950276243094, 0.9904113631045043, 0.9888358108726839, 0.9889089960333207), (0.9776536312849162, 0.9803345713304829, 0.9773724810919342, 0.9776385088219292), (0.9887640449438202, 0.9906072465597778, 0.9886377982577957, 0.9888812740094145), (0.9885057471264368, 0.9902563086272955, 0.9885938256925179, 0.9886429464312937), (0.9883720930232558, 0.9900283937263386, 0.9884847665404723, 0.9884617984773474), (0.9824561403508771, 0.9845901302965016, 0.9823991427561757, 0.9824607342835764), (0.9760479041916168, 0.9786357140295969, 0.9760406087273309, 0.9760139030273048), (0.9757575757575757, 0.9786430539157812, 0.9755085685664199, 0.9757211309144433), (0.968944099378882, 0.9714677719692959, 0.9693347301850059, 0.9687279189566934), (0.9746835443037974, 0.9773062660780976, 0.9749124453498752, 0.9746684787515998), (0.9681528662420382, 0.9713913454795438, 0.9682556074296104, 0.9680973097054665), (0.974025974025974, 0.9774570512974966, 0.9735718826627917, 0.973980027668464), (0.9736842105263158, 0.9765037593984963, 0.9739714492571141, 0.9736369992272365), (0.9727891156462585, 0.9759039146794249, 0.9731237340566787, 0.9728328374607338), (0.9795918367346939, 0.9821513115973758, 0.9800652861877353, 0.9797256769452566), (0.9863945578231292, 0.9874100632118126, 0.9867932520993745, 0.9865207653313486), (0.9862068965517241, 0.9875193534170943, 0.9862032379035947, 0.9862457298734), (0.9791666666666666, 0.9813566545337379, 0.9787125450102881, 0.9790977315592639), (0.9790209790209791, 0.9824591761116774, 0.9785401079107372, 0.9789975582922742), (0.9787234042553191, 0.9819006421541505, 0.9786814881880522, 0.9787637278823648), (0.9856115107913669, 0.9879923399409969, 0.9852405845108085, 0.9853997769737121), (0.9854014598540146, 0.9880654270339388, 0.9855790576660096, 0.9856242643817901), (0.9776119402985075, 0.9809070320041584, 0.9779460904433059, 0.9777193614254255), (0.9774436090225563, 0.98025223688065, 0.9781502628752332, 0.977418001897966), (0.9849624060150376, 0.9859969472553564, 0.9854005314036972, 0.9848705219035175), (0.9849624060150376, 0.9859969472553564, 0.9854005314036972, 0.9848705219035175), (0.9846153846153847, 0.986025641025641, 0.9846153846153847, 0.9844522373126492), (0.9921875, 0.99285888671875, 0.9922096946022727, 0.9921608831649436), (0.984, 0.9838810505050505, 0.9838810505050505, 0.9838810505050505), (0.983739837398374, 0.9834191107986467, 0.9834191107986467, 0.9834191107986467), (0.9917355371900827, 0.9922212356472312, 0.991928022800231, 0.991653652103557), (0.9915966386554622, 0.9924675752654003, 0.9913565426170469, 0.9914630576324501), (0.9827586206896551, 0.9860879904875148, 0.9821640903686087, 0.982362267142291), (0.9827586206896551, 0.9860879904875148, 0.9821640903686087, 0.982362267142291), (0.9826086956521739, 0.9858449905482042, 0.9822155009451796, 0.9822558286074354)]
# # y4 = [(0.7325473161650636, 0.7359787532629776, 0.7325542310953856, 0.7326348710240227), (0.776794035414725, 0.7800868555785909, 0.7767978342275231, 0.7768473896636631), (0.8117564730580826, 0.8147434044080895, 0.811756083567042, 0.8114796817596995), (0.8240971357409713, 0.8278480615120262, 0.8240869546327607, 0.8241423872465248), (0.8684824902723736, 0.8728694241030643, 0.8684450730084582, 0.86857931575631), (0.8618913857677902, 0.8681074023581642, 0.8618187919968424, 0.8617967557723376), (0.8797156916347731, 0.8834252491729931, 0.8796826213850564, 0.8797547199172617), (0.8894440974391006, 0.8917223441333981, 0.8894422384055344, 0.8893499580665024), (0.9, 0.9050503148663007, 0.900053724429058, 0.8997261923605012), (0.9053948397185301, 0.9096214670837097, 0.9053030489659782, 0.9051886764174658), (0.9154443485763589, 0.9188791679223455, 0.9153653285552894, 0.9151011780501588), (0.9295112781954887, 0.9327096285056129, 0.929557555664263, 0.9292904250410986), (0.923469387755102, 0.9286253588545882, 0.9233295177971051, 0.9231415616410144), (0.9260485651214128, 0.9333267903711225, 0.9260257165756657, 0.9253378260190205), (0.9409681227863046, 0.9453806507661787, 0.9410293253265085, 0.9408159427973946), (0.9293820933165196, 0.9338160370344383, 0.929365768392752, 0.9290254319546426), (0.9424364123159303, 0.9487797665037844, 0.9424622430759109, 0.9421645095691908), (0.9416785206258891, 0.9480961615766449, 0.941617233230177, 0.9413398909518164), (0.9356287425149701, 0.940240299790141, 0.9356302151752293, 0.9353137675978749), (0.9415481832543444, 0.9467784757689669, 0.9416092447425976, 0.9411878229290209), (0.9466666666666667, 0.9525692920633548, 0.9466035559246954, 0.9460346593762985), (0.9458041958041958, 0.9498011269831861, 0.9456894851281893, 0.9455061082339646), (0.9543795620437956, 0.9596267582201925, 0.9542521800131422, 0.9539741497064139), (0.9467680608365019, 0.951014137286134, 0.9467018787699429, 0.946488283218363), (0.9482071713147411, 0.9539189538360295, 0.9480706387069847, 0.9476629499371984), (0.9504132231404959, 0.9556406199080425, 0.9506862390999526, 0.9499731860036242), (0.9612068965517241, 0.9652014423306309, 0.9612099856558893, 0.9610120226453721), (0.9642857142857143, 0.9679652964439355, 0.964066345335387, 0.964013336150203), (0.9582366589327146, 0.9634846768212875, 0.9582105984903996, 0.9577275341531427), (0.9641148325358851, 0.9681316813460338, 0.9640500290710231, 0.9637944717482163), (0.9529702970297029, 0.957997906602724, 0.9527960013494553, 0.9525145830140204), (0.9564102564102565, 0.9616915269222961, 0.956076169299994, 0.9560230518892028), (0.9573333333333334, 0.9629474424987199, 0.9570355210287405, 0.9568303545879736), (0.9536784741144414, 0.959501714993159, 0.9539740175635197, 0.9532300621920978), (0.9464788732394366, 0.9527941882312166, 0.9467013870874217, 0.9458143801035848), (0.9507246376811594, 0.9573035230865166, 0.9507253841678193, 0.9500814104665215), (0.9553571428571429, 0.9609964413052088, 0.9553752388097233, 0.9548296029564616), (0.9603658536585366, 0.9654224501963368, 0.9603767203499184, 0.9598832542731911), (0.9529780564263323, 0.9591888867184605, 0.9529932780528912, 0.952278726001297), (0.9615384615384616, 0.9660929618297305, 0.9613275524333218, 0.9612327476497458), (0.956953642384106, 0.9619629857083146, 0.9569734068396409, 0.9567234094558748), (0.95578231292517, 0.9613196431386172, 0.9559275896947701, 0.9555465141182806), (0.9618055555555556, 0.9660976195453098, 0.9617748958298004, 0.9614963882637141), (0.9537366548042705, 0.9594230907241128, 0.9536921291590318, 0.9533854319196353), (0.9562043795620438, 0.9620349410861693, 0.9564012964896621, 0.9558140554554967), (0.9555555555555556, 0.9608647621319655, 0.9561317331204608, 0.9554461474420497), (0.9618320610687023, 0.9661643053399798, 0.9617908162868584, 0.9616925027202022), (0.96875, 0.9715520599210148, 0.968772362331089, 0.9686150370588791), (0.9556451612903226, 0.9612149908691952, 0.9559706394264261, 0.9553479652752416), (0.9508196721311475, 0.9570075675379153, 0.9506961874742811, 0.9502217760364912), (0.966804979253112, 0.9699378999809155, 0.9667815010698224, 0.9665892856931605), (0.9703389830508474, 0.9726783945824056, 0.970195651009222, 0.9701416462991967), (0.9612068965517241, 0.966067814240875, 0.9608952653573685, 0.9607756042897131), (0.9649122807017544, 0.9679810668239801, 0.9650539326785863, 0.9647829594420135), (0.9732142857142857, 0.9758175666099774, 0.97334556960774, 0.9731769613914923), (0.9587155963302753, 0.9657768125126602, 0.9587609535299124, 0.957819259060915), (0.9674418604651163, 0.9709456538301182, 0.9671700018027763, 0.967214020067935), (0.9619047619047619, 0.9659318475928385, 0.9616762768599504, 0.961413503843091), (0.9509803921568627, 0.957781838596612, 0.9504839408530297, 0.9501178948860549), (0.9558823529411765, 0.9630228453036989, 0.9553015323776569, 0.9551044529844908), (0.9554455445544554, 0.9611893897908321, 0.9552016695970288, 0.954953111906013), (0.9744897959183674, 0.976890947705613, 0.9742072828514676, 0.9744062328527135), (0.9583333333333334, 0.9630393433415034, 0.958360087449443, 0.9578693370349715), (0.9576719576719577, 0.9636662404475183, 0.9581408695165309, 0.9569148508101866), (0.9679144385026738, 0.9712040748696745, 0.9677599060266565, 0.9675897857888799), (0.9558011049723757, 0.9606603544625932, 0.9556160037067214, 0.9553549956950194), (0.9553072625698324, 0.9627435987347094, 0.9547848349750598, 0.9542765023213242), (0.9550561797752809, 0.9625132060678145, 0.9545773932649011, 0.954023183349601), (0.9597701149425287, 0.9651765036176976, 0.9592843066014887, 0.9588754078859738), (0.9593023255813954, 0.9648630122005536, 0.9588037452677122, 0.9584285123566048), (0.9590643274853801, 0.9646218444287535, 0.9587280416766413, 0.9582631812419663), (0.9580838323353293, 0.9638382772521681, 0.9578450999082789, 0.9572656182944542), (0.9757575757575757, 0.9775918223190949, 0.9761034787067846, 0.9758586361460926), (0.9751552795031055, 0.9772766307545059, 0.975423159713889, 0.974929097180556), (0.9556962025316456, 0.9617063021300123, 0.9562856227252958, 0.9552776521500413), (0.9681528662420382, 0.97236211798263, 0.9678572877485611, 0.9678042505715448), (0.9675324675324676, 0.9722887502108281, 0.9669940449161227, 0.9672620605462394), (0.9671052631578947, 0.9712789374752672, 0.9673492193402167, 0.966861671306789), (0.9863945578231292, 0.9874421011447251, 0.9863304819573042, 0.9863031172266359), (0.9727891156462585, 0.9757802935645501, 0.9733729179793321, 0.9728250357213436), (0.9659863945578231, 0.9695328966465993, 0.9666296959091629, 0.9659546789967595), (0.9655172413793104, 0.9693724586946941, 0.9655203773634213, 0.9651573034693505), (0.9791666666666666, 0.9813566545337379, 0.9787125450102881, 0.9790977315592639), (0.965034965034965, 0.9694125132532878, 0.9646099820924996, 0.9648074558272078), (0.9716312056737588, 0.9752804829762728, 0.9716798283117886, 0.9716361609644665), (0.9640287769784173, 0.9689656133421983, 0.9636820040370581, 0.9633752186574721), (0.9562043795620438, 0.9611252050913194, 0.9556786900385387, 0.955661761691279), (0.9477611940298507, 0.9544941766483314, 0.9479319818816366, 0.9473300601643089), (0.9473684210526315, 0.9532364746452598, 0.9483002619330281, 0.9467865111812885), (0.9548872180451128, 0.9591854129250251, 0.9555816232310099, 0.9543760226401687), (0.9548872180451128, 0.9591854129250251, 0.9555816232310099, 0.9543760226401687), (0.9615384615384616, 0.9648717948717948, 0.9615384615384616, 0.9611982534448545), (0.953125, 0.958251953125, 0.9540220318418561, 0.952967698962687), (0.96, 0.9646208000000002, 0.9602917171717172, 0.9594286378303348), (0.959349593495935, 0.9627009068809583, 0.9604261098659275, 0.9593748061171696), (0.9586776859504132, 0.9631276220990038, 0.9600653588683777, 0.9586287884655413), (0.9411764705882353, 0.94974513869719, 0.9416142927759339, 0.9407245144711488), (0.9482758620689655, 0.9550102691600908, 0.9467972333106845, 0.946969310911047), (0.9482758620689655, 0.9550102691600908, 0.9467972333106845, 0.946969310911047), (0.9391304347826087, 0.9508991235607493, 0.9378052930056712, 0.93639594432033)]
# # y5 = [(0.726962457337884, 0.7306668136861687, 0.7269697068249427, 0.7273159112484697), (0.775085430257844, 0.7794306736566737, 0.7750908259158135, 0.775364790855555), (0.8098903662234663, 0.8130716972884177, 0.8098912394334906, 0.810026023266363), (0.8166251556662516, 0.8203351974852506, 0.8165990365501073, 0.8169386206101059), (0.8529182879377432, 0.8582134637547324, 0.8528793077927856, 0.8531011327537229), (0.851123595505618, 0.8579106431207102, 0.8510499560814673, 0.8514100643687017), (0.8731547293603061, 0.8773400783758083, 0.8731238251693572, 0.8734918719701882), (0.8732042473454091, 0.8758449344979161, 0.8732007114188154, 0.8735505591077275), (0.8908450704225352, 0.8959106123775725, 0.8908973782967087, 0.8908320711706332), (0.8952306489444879, 0.9012560757297191, 0.8951149151649617, 0.895003556931299), (0.911993097497843, 0.9158960139026564, 0.9119378523673392, 0.9118013116722702), (0.924812030075188, 0.9281262038526982, 0.9248556242993814, 0.9247609268452932), (0.9193877551020408, 0.9249537865016826, 0.9192578453102415, 0.9191166700749883), (0.9249448123620309, 0.9304764746305294, 0.9249190601415567, 0.92447317407186), (0.9315230224321134, 0.937435091195101, 0.9315993075886931, 0.9314054473175291), (0.9180327868852459, 0.923420313191142, 0.9180020335392471, 0.9176449328995667), (0.9410977242302544, 0.9463469550815532, 0.9411133222644549, 0.9408513416329474), (0.930298719772404, 0.9382593985841892, 0.9302269535158031, 0.9297308390927412), (0.9341317365269461, 0.9379248608112001, 0.9341450127839213, 0.9337469574412868), (0.9289099526066351, 0.9355479044031974, 0.9289983004275135, 0.9285103087943414), (0.9433333333333334, 0.948872403146011, 0.9432513755471181, 0.9427383383107715), (0.9335664335664335, 0.9386940055668153, 0.9334303477893395, 0.9331805380009263), (0.9379562043795621, 0.944376589528987, 0.9378793116051419, 0.9372429874551748), (0.9372623574144486, 0.9419283615928328, 0.9372561797238599, 0.9368617743082972), (0.9362549800796812, 0.9446270027039623, 0.9360934047763102, 0.9354622979302337), (0.9380165289256198, 0.9431974323730182, 0.9383817266607475, 0.9375584102650142), (0.9547413793103449, 0.960264232815579, 0.9547935162770453, 0.9544735334369034), (0.9553571428571429, 0.960646318761379, 0.9550899382560484, 0.9547462873431095), (0.951276102088167, 0.9581120785530857, 0.9512078727657678, 0.9505327763828969), (0.9569377990430622, 0.961681179001281, 0.9568654927487676, 0.9566414767613902), (0.9405940594059405, 0.9467682951363127, 0.9404099640475356, 0.9400674154249543), (0.9512820512820512, 0.958340168262723, 0.9508754835303981, 0.9505201844563002), (0.952, 0.9589861857708436, 0.9516709116773077, 0.9510601890819104), (0.9564032697547684, 0.9615022724193695, 0.9567000107056663, 0.9560198201802784), (0.9436619718309859, 0.9493402350937651, 0.9438469725965912, 0.9430529440788806), (0.9449275362318841, 0.9534389561751253, 0.9449534150798996, 0.9439695093490322), (0.9613095238095238, 0.9659428053354363, 0.9612931448405364, 0.9609710545069096), (0.9603658536585366, 0.9650247357705939, 0.9603666020532599, 0.9599249617550372), (0.9529780564263323, 0.9591958091494266, 0.953034605610569, 0.952312544770835), (0.9519230769230769, 0.958676397873366, 0.951752054035708, 0.9514446689507031), (0.9503311258278145, 0.9560506480901685, 0.9504400296865539, 0.949641144474372), (0.9523809523809523, 0.9589717527909948, 0.9526814224231966, 0.9517455758852138), (0.9583333333333334, 0.9644236448895644, 0.9583044820566522, 0.9575259996499381), (0.9537366548042705, 0.9601245280219711, 0.9535733496682908, 0.9530720553563652), (0.9562043795620438, 0.9620765540516366, 0.9563762417868945, 0.9558519607725131), (0.9555555555555556, 0.9611891169008722, 0.9559945589092127, 0.9553654263332791), (0.9541984732824428, 0.9601858512871857, 0.9542053622568958, 0.9535908713465655), (0.95703125, 0.9622730655426993, 0.9567755989040385, 0.9563667877028081), (0.9596774193548387, 0.964780350047975, 0.9598915431156855, 0.9594569686704998), (0.9467213114754098, 0.9553384044874031, 0.9465184248982386, 0.9452131765012334), (0.9585062240663901, 0.9631759164405851, 0.9583559636933373, 0.9578896220236547), (0.9576271186440678, 0.9619864199980509, 0.9573198067584612, 0.9569613317455855), (0.9525862068965517, 0.9590331697186002, 0.9522759091284542, 0.9519312242689062), (0.9473684210526315, 0.9560232624944589, 0.9475485464058871, 0.946476762796199), (0.96875, 0.9729586442189657, 0.9688211753002636, 0.9684820625668841), (0.9587155963302753, 0.9653434580634954, 0.9587307933301535, 0.9578140347223096), (0.9534883720930233, 0.9608029952342788, 0.9526554894537587, 0.9520435454550467), (0.9523809523809523, 0.9601570082578487, 0.9520832523485585, 0.9510149313356914), (0.9509803921568627, 0.9587587127500925, 0.9503809584836113, 0.9500107620989704), (0.9509803921568627, 0.9600165922974458, 0.9502820954089697, 0.9496255212225598), (0.9504950495049505, 0.958083830790583, 0.9502609775080179, 0.9495271090316193), (0.9642857142857143, 0.9689530786804519, 0.9635215356887317, 0.9636414840695533), (0.953125, 0.9600664210311317, 0.9532447600684906, 0.9522764555450677), (0.9523809523809523, 0.9592719816008612, 0.9528218694885361, 0.9515216123326034), (0.9572192513368984, 0.963650281315065, 0.9569503318323702, 0.9560293632113882), (0.9502762430939227, 0.9588024306916316, 0.9500339091430634, 0.9491677675537836), (0.9497206703910615, 0.9581653653315513, 0.9491865390417246, 0.9483535635328836), (0.949438202247191, 0.9579443060106322, 0.9489160184384905, 0.9480669147604551), (0.9482758620689655, 0.956768643512366, 0.9475134319813274, 0.9462636698880056), (0.9418604651162791, 0.9516120266386707, 0.941088651523346, 0.9397566220511056), (0.9473684210526315, 0.956034025217077, 0.9467927453461464, 0.9454368898501725), (0.9461077844311377, 0.9551474166345516, 0.9456995488764287, 0.9441959307596929), (0.9393939393939394, 0.950118411000764, 0.9404063062740747, 0.9384520077412639), (0.9503105590062112, 0.9570377547558211, 0.9505998161914215, 0.9493798363662549), (0.9367088607594937, 0.9460954681873983, 0.9373040034792959, 0.9348960631559117), (0.9426751592356688, 0.9523694886943128, 0.9417768324417679, 0.9400085141907524), (0.9415584415584416, 0.9504660670751074, 0.9408272617920113, 0.9397295460951958), (0.9539473684210527, 0.9592300877792569, 0.9539934615246249, 0.9527914079290464), (0.9523809523809523, 0.9579636430948384, 0.9531585079273543, 0.9516179629107722), (0.9523809523809523, 0.9588597343699385, 0.952960177866467, 0.9520921974760652), (0.9659863945578231, 0.9695328966465993, 0.9666296959091629, 0.9659546789967595), (0.9586206896551724, 0.9638959851088271, 0.958528700787916, 0.9580261183921113), (0.9513888888888888, 0.9589025862897391, 0.9506322108318636, 0.9506789602340818), (0.951048951048951, 0.9594996719257074, 0.9505261638128772, 0.9501326694864404), (0.950354609929078, 0.9585561563832276, 0.950645148155334, 0.9496121814379659), (0.9424460431654677, 0.9502775057352977, 0.9424388957487657, 0.9409416679128616), (0.9562043795620438, 0.9621552724008902, 0.9556786900385387, 0.9554639513405788), (0.9477611940298507, 0.9544941766483314, 0.9479319818816366, 0.9473300601643089), (0.9473684210526315, 0.9544735043350002, 0.9485625090294659, 0.9469538523202431), (0.9624060150375939, 0.9661671529894719, 0.9630941388559118, 0.9619958837846747), (0.9624060150375939, 0.9661671529894719, 0.9630941388559118, 0.9619958837846747), (0.9615384615384616, 0.9659566074950691, 0.9615384615384616, 0.9609957811515896), (0.9453125, 0.9531813401442308, 0.9466978130918561, 0.9448973073341049), (0.952, 0.9595254153846154, 0.9523797171717172, 0.9507223361320332), (0.9349593495934959, 0.9451969472790412, 0.9367313995263629, 0.9341309091139324), (0.9338842975206612, 0.9451912748135683, 0.9358734017312667, 0.9327955453390875), (0.9411764705882353, 0.9519656073042133, 0.9416142927759339, 0.9400146577482987), (0.9224137931034483, 0.9368209673798258, 0.9198841196704604, 0.9185537605133842), (0.9396551724137931, 0.9513464240872088, 0.9369118396466791, 0.9362279759135211), (0.9304347826086956, 0.9438257432548548, 0.9285897920604915, 0.926781070221524)]

# y0 = [0.9426229508196722, 0.9426229508196722, 0.9672131147540983, 0.9467213114754098, 0.9549180327868853, 0.9631147540983607, 0.9549180327868853, 0.9549180327868853, 0.9754098360655737, 0.9631147540983607, 0.9795081967213115, 0.9713114754098361, 0.9713114754098361, 0.9672131147540983, 0.9713114754098361, 0.9836065573770492, 0.9713114754098361, 0.9795081967213115, 0.9713114754098361, 0.9754098360655737, 0.9713114754098361, 0.9713114754098361, 0.9836065573770492, 0.9713114754098361, 0.9713114754098361, 0.9713114754098361, 0.9713114754098361, 0.9713114754098361, 0.9713114754098361, 0.9795081967213115, 0.9795081967213115, 0.9795081967213115, 0.9795081967213115, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9836065573770492, 0.9795081967213115, 0.9795081967213115, 0.9754098360655737, 0.9754098360655737, 0.9754098360655737, 0.9754098360655737, 0.9754098360655737, 0.9836065573770492, 0.9877049180327869, 0.9877049180327869, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9918032786885246, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9877049180327869, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623, 0.9959016393442623]

# # y0 = [i[0] for i in y0]
# # y1 = [i[0] for i in y1]
# # y2 = [i[0] for i in y2]
# # y3 = [i[0] for i in y3]
# # y4 = [i[0] for i in y4]
# # y5 = [i[0] for i in y5]

# # linear = [0.9183673469387755, 0.9047619047619048, 0.8843537414965986, 0.8843537414965986]
# # polynom = [0.7210884353741497, 0.8707482993197279, 0.8163265306122449, 0.8095238095238095]
# # sigmoid = [0.891156462585034, 0.8775510204081632, 0.8435374149659864, 0.8435374149659864]
# # rbf = [0.8979591836734694, 0.8299319727891157, 0.8095238095238095, 0.8027210884353742]

# # cs = [1, 10, 100, 1000]
# # legend_dict = {'1': 'o', '10': 's', '100': '^', '1000': 'D'}

# fig = plt.figure(figsize=(6,2))
# ax = fig.add_subplot(111)

# ys0 = lowess(y0, x)
# lowess_x0 = ys0[:,0]
# lowess_y0 = ys0[:,1]

# # ys1 = lowess(y1, x)
# # lowess_x1 = ys1[:,0]
# # lowess_y1 = ys1[:,1]

# # ys2 = lowess(y2, x)
# # lowess_x2 = ys2[:,0]
# # lowess_y2 = ys2[:,1]

# # ys3 = lowess(y3, x)
# # lowess_x3 = ys3[:,0]
# # lowess_y3 = ys3[:,1]

# # ys4 = lowess(y4, x)
# # lowess_x4 = ys4[:,0]
# # lowess_y4 = ys4[:,1]

# # ys5 = lowess(y5, x)
# # lowess_x5 = ys5[:,0]
# # lowess_y5 = ys5[:,1]

# # # legend_dictionary = {'Alanus-de-Insulis': 'Alan of Lille'}

# for p1, p2 in zip(x,y0):
# 	ax.scatter(p1, p2, marker='o', color='w', s=2, edgecolors='k', linewidth=0.2)
# # for p1, p2 in zip(x,y1):
# # 	ax.scatter(p1, p2, marker='o', color='w', s=7, edgecolors='k', linewidth=0.5)
# # for p1, p2 in zip(x,y2):
# # 	ax.scatter(p1, p2, marker='o', color='w', s=7, edgecolors='#EE220C', linewidth=0.5)
# # for p1, p2 in zip(x,y3):
# # 	ax.scatter(p1, p2, marker='o', color='w', s=7, edgecolors='#61D836', linewidth=0.5)
# # for p1, p2 in zip(x,y4):
# # 	ax.scatter(p1, p2, marker='o', color='w', s=7, edgecolors='#00A2FF', linewidth=0.5)
# # for p1, p2 in zip(x,y5):
# # 	ax.scatter(p1, p2, marker='o', color='w', s=7, edgecolors='#F8BA00', linewidth=0.5)

# ax.plot(lowess_x0, lowess_y0, color='k', linewidth=1.0, markersize=1.5, linestyle='--')
# # ax.plot(lowess_x1, lowess_y1, color='k', linewidth=1.0, markersize=2, linestyle='--')
# # ax.plot(lowess_x2, lowess_y2, color='#EE220C', linewidth=1.0, markersize=2, linestyle='--')
# # ax.plot(lowess_x3, lowess_y3, color='#61D836', linewidth=1.0, markersize=2, linestyle='--')
# # ax.plot(lowess_x4, lowess_y4, color='#00A2FF', linewidth=1.0, markersize=2, linestyle='--')
# # ax.plot(lowess_x5, lowess_y5, color='#F8BA00', linewidth=1.0, markersize=2, linestyle='--')

# # collected_patches = []
# # for feat_type in legend_dictionary:
# # 	# legend_patch = mlines.Line2D([0], [0], markerfacecolor='k', color=legend_dictionary[feat_type], markersize=5)
# # 	legend_patch = mpatches.Patch(color=customized_colors[feat_type], label=legend_dictionary[feat_type])
# # 	collected_patches.append(legend_patch)
# # plt.legend(handles=collected_patches, fontsize=7, fancybox=True)

# # # ax.yaxis.grid(linestyle='dashed', color='k', linewidth=0.3)
# # # ax.plot(lowess_x, lowess_y, color='k', linewidth=0.7, markersize=2, linestyle='--')

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Alfios']

# ax.set_xlabel('number of folds')
# ax.set_ylabel('accuracy')

# # # ax.set_xticklabels(labels=['linear', 'polynomial', 'sigmoid', 'radial'])
# ax.set_xticks([2, 50, 100, 150, 200, len(train_texts)])
# # ax.set_yticks([0.85, 0.90, 0.95, 1.0])
# # ax.set_ylim(0.96, 1)

# for tick in ax.xaxis.get_major_ticks():
# 	tick.label.set_fontsize(7)
# for tick in ax.yaxis.get_major_ticks():
# 	tick.label.set_fontsize(7)

# # Despine
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(True)
# ax.spines['bottom'].set_visible(True)

# plt.tight_layout()
# plt.show()

# fig.savefig("/Users/jedgusse/compstyl/output/fig_output/diff-cval-splits", \
# 			transparent=True, format='pdf')


