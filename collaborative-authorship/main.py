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
from preprocess import DataReader, Vectorizer, windower
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
import click
import cmd
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

# Fetch local directory names of user
code_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = code_dir.split('/')[:-1]
data_dir.append('data')
data_dir = '/'.join(data_dir)

varia_dir = code_dir.split('/')[:-1]
varia_dir.append('varia')
varia_dir = '/'.join(varia_dir)

"""
Parameters
----------
sample_size: determines the size (number of words) of 
			 the discrete text segments
n_feats: determines vector length (i.e. number of 
		 features taken into account)
step_size: determines size of overlapping 
		   window if rolling sampling instead of 
		   discrete sampling
rnd_dict: dictionary by which to make random samples
		  samples or subsets are randomly sampled (i.e.
		  by chance) from a population in n iterative turns
invalid_words: enter tokens which you want to exclude from the analysis
			   e.g. if the content word 'deus' (Lat.) is considered to be 
			   irrelevant in the analysis, enter the string in the list
function_words_only: generates a list of function words form the varia directory
test_dict: list of test set items. 
		   For instance: if you have a training corpus that is randomly sampled, which 
		   needs to be benchmarked against 'true', real-life test data.
"""

invalid_words = open(varia_dir + '/invalid_words.txt').read().split()
function_words_only = open(varia_dir + '/fword_list.txt').read().split()

@click.command()
@click.option('--train_path', prompt='Which authors are training corpus? Type author name as in data directory')
@click.option('--test_path', prompt='Which authors are test corpus? Type - if none, or author name as in data directory')
@click.option('--feat_type', prompt='Enter feature type (function word, word ngram, word, char ngram)', default='function word')
@click.option('--feat_number', prompt='Enter feature number', default=250)
@click.option('--sample_len', prompt='Enter sample length', default=2000)
@click.option('--sample_type', prompt='Enter sample type', default='discrete')
@click.option('--vectorization_method', prompt='Enter vectorization method', default='tfidf')
@click.option('--scaling_method', prompt='Enter scaling method', default='standard_scaler')
@click.option('--analysis', prompt='Enter method of analysis', default='PCA')

def params_and_go(train_path, test_path,
				  feat_type, feat_number, sample_len, sample_type, 
				  vectorization_method, scaling_method, analysis):
	"""
	Function that takes all input parameters.
	
	Enter feature type: [char ngrams, word ngrams, function words, tokens]
	Enter number of features: [...] (at will)
	Enter sample length: [...] (at will)
	Enter sampling method: [random, rolling, discrete]

	Enter vectorization method: [raw, tfidf]
	Enter scaling method: [standard]

	Enter method of analysis: [PCA, SVM, network analysis, impostors method]
	"""

	"""
	Step 1	
	Sampling and tokenization
	"""
	train_path = data_dir + '/' + train_path
	test_path = data_dir + '/' + test_path

	test_dict = {'': ''}

	if sample_type == 'random':
		# Random sampling method
		n_samps = input("Enter number of random samples: ")
		n_samps = int(n_samps)
		rnd_dct = {'n_samples': n_samps,
				   'smooth_train': True,
				   'smooth_test': False}

		authors, titles, texts = DataReader(train_path, sample_len,
									test_dict, rnd_dct
									).metadata(sampling=False,
									type='folder',
									randomization=True)

	elif sample_type == 'rolling':
		# Rolling sampling method (shingling, making windows)
		rnd_dct = {'n_samples': 0,
				   'smooth_train': True,
				   'smooth_test': False}
		step_size = input("Enter step size: ")
		step_size = int(step_size)

		authors, titles, texts = windower(train_path, step_size, sample_len, rnd_dct)

	elif sample_type == 'discrete':
		# Discrete (normal) sampling method

		rnd_dct = {'n_samples': 0,
				   'smooth_train': True,
				   'smooth_test': False}
		authors, titles, texts = DataReader(train_path, sample_len,
									test_dict, rnd_dct
									).metadata(sampling=True,
									type='folder',
									randomization=False)
	"""
	Step 2
	Feature extraction and vectorization
	"""

	scaler = scaling_method
	if feat_type == 'function word':
		analyzer = 'word'
		vocab = function_words_only
		ngram = 1
	elif feat_type == 'word ngram':
		analyzer = 'word'
		vocab = None
		ngram = input("Enter number of ngrams (bi-, tri-, 4-...grams): ")
		ngram = int(ngram)
	elif feat_type == 'word':
		analyzer = 'word'
		vocab = None
		ngram = 1
	elif feat_type == 'char ngram':
		analyzer = 'char'
		vocab = None
		ngram = input("Enter number of ngrams (bi-, tri-, 4-...grams): ")
		ngram = int(ngram)

	if vectorization_method == 'raw':
		vectors, features, scaling_model = Vectorizer(texts, invalid_words,
										  n_feats=feat_number,
										  feat_scaling=scaler,
										  analyzer=analyzer,
										  vocab=vocab,
										  ngram=ngram
										  ).raw()

		print("::: These are your corpus's top 10 most frequent features :::")
		print(features[:10])

	elif vectorization_method == 'tfidf':
		vectors, features, scaling_model = Vectorizer(texts, invalid_words,
													  n_feats=feat_number,
													  feat_scaling=scaler,
													  analyzer=analyzer,
													  vocab=vocab,
													  ngram=ngram
													  ).tfidf(smoothing=False)
		print("::: These are your corpus's top 10 most frequent features :::")
		print(features[:10])

if __name__ == '__main__':
	params_and_go()

	# # # LexicalRichness(desired_authors, desired_titles, desired_texts).plot(split_size)

	# grid_vocab, grid_nfeats = PipeGridClassifier(authors, titles, texts, 
	# 											   n_feats, test_dict, invalid_words
	# 											   ).fit_transform_classify(uisualize_db=False)

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


