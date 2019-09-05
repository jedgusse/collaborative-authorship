#!/usr/bin/env

from cltk.stem.latin.declension import CollatinusDecliner
from collatex import *
from collections import Counter
from difflib import SequenceMatcher
from itertools import compress, combinations, zip_longest, groupby
from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager as font_manager, rcParams
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from preprocess import DataReader, Vectorizer
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_regression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, LabelBinarizer, MinMaxScaler
from sklearn.utils.fixes import signature
from statsmodels.nonparametric.smoothers_lowess import lowess
from string import punctuation
from tqdm import tqdm, trange
from visualization import PrinCompAnal, GephiNetworks, RollingDelta, HeatMap, IntrinsicPlagiarism, LexicalRichness
import argparse
import glob
import Levenshtein
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import operator
import os
import pandas as pd
import pickle
import random
import re
import scipy
import seaborn.apionly as sns
import sys

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
		text_samples = [text[i:i+sample_size] for i in range(0, len(text), sample_size)]
		text_samples = [i for i in text_samples if len(i) == sample_size]
		# Append to training set
		for idx, sample in enumerate(text_samples):
			sample = ' '.join(sample)
			sample_title = '{}_{}'.format(title, str(idx))
			d[author].append((sample_title, sample))

	return d

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

"""
Parameters
----------
"""

sample_size = 1100
n_feats = 350
step_size = 100

rnd_dct = {'n_samples': 800,
		   'smooth_train': True,
		   'smooth_test': False}

# Enter tokens which you want to exclude from the analysis
invalid_words = []
function_words_only = open('/Users/jedgusse/compstyl/params/fword_list.txt').read().split()
test_dict = {'derolez': 'test-ms'}

# if __name__ == '__main__':

	# authors, titles, texts = DataReader(folder_location, sample_size,
	# 									test_dict, rnd_dct
	# 									).metadata(sampling=True,
	# 									type='folder',
	# 									randomization=False)

	# for title, samp in zip(titles, texts):
	# 	sample_n = int(title.split('_')[-1].split('-')[-1])
	# 	if sample_n in [6]:
	# 		print(sample_n)
	# 		print(samp[:20])
	# 		print()

	# """
	# Test data
	# ---------
	# """
	# dir_location = 'hildegard-vita/test_set'
	# test_data = load_and_split(dir_location)

	# """
	# shingling
	# uncheck for shingling
	# """
	# test_set_location = '/Users/jedgusse/compstyl/impostors-method/hildegard-vita/test_set/Vita-Hildegardis_Vita-Hildegardis.txt'
	# original_test_txt = open(test_set_location).read()
	# test_txt = re.sub('[%s]' % re.escape(punctuation), '', original_test_txt) # Escape punctuation and make characters lowercase
	# text_txt = re.sub('\d+', '', test_txt)
	# test_txt = test_txt.lower().split()

	# # Collect test samples in dictionary by keyword of range
	# test_data = {}
	# # "shingling": make windows	
	# steps = np.arange(0, len(test_txt), step_size)

	# step_ranges = []
	# test_data = {}
	# key = dir_location.split('/')[-1].split('.')[0].split('_')[0]
	# test_data[key] = []
	# for each_begin in steps:
	# 	sample_range = range(each_begin, each_begin + sample_size)
	# 	step_ranges.append(sample_range)
	# 	text_sample = []
	# 	for index, word in enumerate(test_txt):
	# 		if index in sample_range:
	# 			text_sample.append(word)
	# 	title = '{}-{}'.format(str(each_begin), str(each_begin + sample_size))
	# 	if len(text_sample) == sample_size:
	# 		value = (title, ' '.join(text_sample))
	# 		test_data[key].append(value)

	# for key in test_data.keys():
	# 	test_titles = [i[0] for i in test_data[key]]
	# 	test_texts = [i[1] for i in test_data[key]]

	# # # LexicalRichness(desired_authors, desired_titles, desired_texts).plot(split_size)

	# grid_vocab, grid_nfeats = PipeGridClassifier(authors, titles, texts, 
	# 											   n_feats, test_dict, invalid_words
	# 											   ).fit_transform_classify(uisualize_db=False)

	# vectors, features, scaling_model = Vectorizer(texts, invalid_words,
	# 								  n_feats=n_feats,
	# 								  feat_scaling='standard_scaler',
	# 								  analyzer='word',
	# 								  vocab=None
	# 								  ).raw()

	# vectors, features, scaling_model = Vectorizer(texts, invalid_words,
	# 											  n_feats=n_feats,
	# 											  feat_scaling='standard_scaler',
	# 											  analyzer='word',
	# 											  vocab=function_words_only
	# 											  ).tfidf(smoothing=False)

	# PrinCompAnal(authors, titles, vectors, features, sample_size, n_components=3, show_pc2_pc3=False).plot(
	# 												show_samples=True,
	# 												show_loadings=True,
	# 												sbrn_plt=False)
	
	# HeatMap(vectors, features, authors, titles).plot()

	# GephiNetworks(folder_location, sample_size, invalid_words).plot(feat_range=[150],
	# 																random_sampling=False,
	# 																corpus_size=90)

	# variances = []
	# for subsample in range(0,150):
	# 	uariance = GephiNetworks(folder_location, sample_size, invalid_words).plot(feat_range=list(range(100, 1100, 100)),
	# 																	random_sampling='simple',
	# 																	corpus_size=90)
	# 	print(uariance)
	# 	variances.append(uariance)
	# print("CORPUS SIZE: {}".format(str(corpsize*1000)))
	# print()
	# print(variances)
	# var_of_it = np.var(variances)
	# mean_of_it = np.mean(variances)
	# print()
	# print("MEAN", mean_of_it)
	# print("CV", var_of_it/mean_of_it)
	# print()

	# LexicalRichness(authors, titles, texts).plot(sample_size)

	# RollingDelta(folder_location, n_feats, invalid_words, sample_size, step_size, test_dict, rnd_dct).plot()

	# IntrinsicPlagiarism(folder_location, n_feats, 
	# 					invalid_words, sample_size, 
	# 					step_size).plot(support_ngrams=True, 
	# 					support_punct=False)
	


