#!/usr/bin/env

from classification import PipeGridClassifier, DimRed
from cltk.stem.latin.declension import CollatinusDecliner
from collatex import *
from collections import Counter
from difflib import SequenceMatcher
from itertools import compress, combinations, zip_longest
from matplotlib import colors
from matplotlib import font_manager as font_manager, rcParams
from preprocess import DataReader, Vectorizer
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_regression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, LabelBinarizer, MinMaxScaler
from string import punctuation
from visualization import PrinCompAnal, GephiNetworks, RollingDelta, HeatMap, IntrinsicPlagiarism, LexicalRichness
import argparse
import glob
import Levenshtein
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn.apionly as sns

"""
FUNCTIONS USED IN SIMULATION OF DECISION BOUNDARY
-------------------------------------------------
"""
def to_dense(X):
        X = X.todense()
        X = np.nan_to_num(X)
        return X

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	x1, _ = ax1.transData.transform((v1, 0))
	x2, _ = ax2.transData.transform((v2, 0))
	inv = ax2.transData.inverted()
	dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
	minx, maxx = ax2.get_xlim()
	ax2.set_xlim(minx+dx, maxx+dx)

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
========================
"""

folder_location = '/Users/jedgusse/compstyl/impostors-method/abelard-heloise/train_set'
# test_set_location = '/Users/jedgusse/compstyl/impostors-method/hildegard-vita/test_set/Vita-Hildegardis_Vita-Hildegardis.txt'

# PARAMETERS
### ||| ------ ||| ###

sample_size = 1200
n_feats = 250
step_size = 100

rnd_dct = {'n_samples': 800,
		   'smooth_train': True,
		   'smooth_test': False}

invalid_words = []
function_words_only = open('/Users/jedgusse/compstyl/params/fword_list.txt').read().split()

test_dict = {'derolez': 'test-ms'}

# For classification tests, split data into training and test corpus (classifier will train and evaluate on training corpus, 
# and predict on new test corpus)

authors, titles, texts = DataReader(folder_location, sample_size,
									test_dict, rnd_dct
									).metadata(sampling=True,
									type='folder',
									randomization=False)

label_dict = {}
inverse_label = {}
for author in authors: 
	label_dict[author.split('_')[0]] = 0 
for i, key in zip(range(len(label_dict)), label_dict.keys()):
	label_dict[key] = i
	inverse_label[i] = key

"""
DISCRETE SAMPLES
comment out if shingled
----------------
"""
# test_authors, test_titles, test_texts = DataReader(test_location, sample_size,
# 									test_dict, rnd_dct
# 									).metadata(sampling=True,
# 									type='folder',
# 									randomization=False)

"""
SHINGLED SAMPLES
comment out if discrete
----------------
"""

# dir_location = 'hildegard-vita/test_set'
# test_data = load_and_split(dir_location)
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
for author in authors:
	label = label_dict[author.split('_')[0]]
	Y_train.append(label)

Y_train = np.array(Y_train)

"""
MAKE 2D CLASSIFIER
------------------
"""
# Load classifier parameters from the gridsearch into this 2D-simulation
# Decision function = 'ovr' (one vs rest) or 'ovo' (one vs one)

model = '/Users/jedgusse/compstyl/output/numpy_output/heloise-abelard-models/try_-tfidf_MFW-400-normal-StratifiedKFold-1-model'
model_name = model.split('/')[-1]
grid = pickle.load(open(model, 'rb'))

best_model = grid.best_estimator_

selected_features = grid.best_params_['vectorizer'].get_feature_names()
vectorizer = best_model.named_steps['vectorizer']
vectorizer = vectorizer.set_params(vocabulary=selected_features)
scaler = best_model.named_steps['feature_scaling']
dim_red = best_model.named_steps['reduce_dim']

X_train = vectorizer.fit_transform(texts).toarray()
# x_test = vectorizer.transform(test_texts).toarray()

"""
SCALING
-------
"""
# Retrieve all original scaling weights from best model
# Apply selected scaling weights to selected features

feature_scaling = {feat: (mean, scale) for mean, scale, feat in 
				   zip(scaler.mean_, \
					   scaler.scale_, \
					   selected_features)}
model_means = []
model_vars = []
for feature in selected_features:
	model_means.append(feature_scaling[feature][0])
	model_vars.append(feature_scaling[feature][1])
model_means = np.array(model_means)
model_vars = np.array(model_vars)

X_train = (X_train - model_means) / model_vars
# x_test = (x_test - model_means) / model_vars

"""
PRINCIPAL COMPONENTS ANALYSIS
-----------------------------
"""
# Visualize decision boundary with PCA
# Unfortunately, we have to instantiate a new SVM model, one that
# refits on data that has become 2dimensional.
# I did not find a way to do have a decision hyperplane become 2d.
# Transform with same PCA values

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
# x_test = pca.transform(x_test)
xx, yy = make_meshgrid(X_train[:,0], X_train[:,1])
var_exp = pca.explained_variance_ratio_
var_pc1 = np.round(var_exp[0]*100, decimals=2)
var_pc2 = np.round(var_exp[1]*100, decimals=2)
explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)
loadings = pca.components_.transpose()
vocab_weights_p1 = sorted(zip(selected_features, loadings[:,0]), \
						  key=lambda tup: tup[1], reverse=True)
vocab_weights_p2 = sorted(zip(selected_features, loadings[:,1]), \
				   		  key=lambda tup: tup[1], reverse=True)

"""
RANK FEATURES
-------------
"""
# Plot only the z features that on average have the highest weight
# Choose number of features per principal component to be plotted (slice from vocab_weights)
# Set threshold as to which 'best discriminators in the PC' may be plotted

z = 50

scaler = MinMaxScaler()
scaled_loadings = scaler.fit_transform(np.abs(loadings))
ranks = scaled_loadings.mean(axis=1)

scaler = MinMaxScaler()
ranks = scaler.fit_transform(ranks.reshape(-1,1))
ranks = ranks.flatten()

high_scorers = []
font_dict_scores = []
for idx, (feature, rank, coords) in enumerate(sorted(zip(selected_features, ranks, loadings), \
							key=lambda tup:tup[1], reverse=True)):
	print(rank)
	font_size = rank * 20
	font_dict_scores.append(font_size)
	if idx in list(range(0,z)):
		high_scorers.append(feature)

# z = 5
# printed_features_p1 = vocab_weights_p1[:z] + vocab_weights_p1[-z:]
# printed_features_p2 = vocab_weights_p2[:z] + vocab_weights_p2[-z:]

print("Explained variance: ", explained_variance)
print("Number of words: ", n_feats)
print("Sample size : ", sample_size)

"""
MAKE 2D CLASSIFIER
------------------
"""
# Load classifier parameters from the gridsearch into this 2D-simulation
# Decision function = 'ovr' (one vs rest) or 'ovo' (one vs one)
best_clf = best_model.named_steps['classifier']
clf_params = best_clf.get_params()
svm_clf = svm.SVC(kernel=clf_params['kernel'], 
				  C=clf_params['C'], 
				  decision_function_shape=clf_params['decision_function_shape'])
svm_clf.fit(X_train, Y_train)

"""
PLOTTING
--------
http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
"""

# FONTS
# Custom fonts should be added to the matplotlibrc parameter file: cd '/Users/jedgusse/.matplotlib'
# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Alfios']

# Concatenation of training and test data
# all_titles = train_titles + test_titles
# all_props = train_props + test_props
# all_coords = np.concatenate((X_train, x_test), axis=0)

x1, x2 = X_train[:,0], X_train[:,1]
# t1, t2 = x_test[:,0], x_test[:,1]

legend_dict = {'Nicolaus-Claraevallensis': 'Nicholas of Montiéramey',
			   'Bernardus-Claraevallensis': 'Bernard of Clairvaux',
			   'Theodericus-Epternacensis': 'Theoderic of Echternach',
			   'Guibertus-Gemblacensis': 'Guibert of Gembloux',
			   'Hild': 'Hildegard of Bingen',
			   'Heloysa': 'Ascribed to Heloise',
			   'Petr-Abel': 'Ascribed to Abelard'}

# Define contour colours (light background)
contour_dict = {'Nicolaus-Claraevallensis': '#BDE5B6',
			   'Bernardus-Claraevallensis': '#9BC1E0',
			   'Theodericus-Epternacensis': '#FDE3B3',
			   'Guibertus-Gemblacensis': '#D1EDFF',
			   'Hild': '#AFDFA8',
			   'Heloysa': '#CBE9D1',
			   'Petr-Abel': '#FCD7D3'}

# Define node colours
colors_dict= {'Nicolaus-Claraevallensis': '#4AB836',
			   'Bernardus-Claraevallensis': '#5F7AC4',
			   'Theodericus-Epternacensis': '#FDA660',
			   'Guibertus-Gemblacensis': '#88BEDC',
			   'Hild': '#8DD08A',
			   'Petr-Abel': '#F77C6F',
 			   'Heloysa': '#53B466'}

"""----------
plot contours
"""
fig = plt.figure(figsize=(4.5,3))
ax = fig.add_subplot(111)

# Fixed list of hex color codes (custom cmap for contours) needs to be made
# This equals making a custom cmap!
# inspiration: https://stackoverflow.com/questions/9707676/defining
# -a-discrete-colormap-for-imshow-in-matplotlib
bounds = []
listed_colors = []
for title, label in label_dict.items():
	idx = label
	bounds.insert(idx, label)
	listed_colors.insert(idx, contour_dict[title])
contour_cmap = colors.ListedColormap(listed_colors)
norm = colors.BoundaryNorm(bounds, contour_cmap.N)

plot_contours(ax, svm_clf, xx, yy, cmap=contour_cmap, alpha=0.8)

"""---------------
plot train samples
"""
for index, (p1, p2, author, title) in enumerate(zip(X_train[:, 0], X_train[:, 1], authors, titles)):
	ax.scatter(p1, p2, color=colors_dict[author], s=15, zorder=10, edgecolors='k', linewidth=0.3)

"""---------------
plot test samples
"""

# for index, (p1, p2, title) in enumerate(zip(x_test[:, 0], x_test[:, 1], test_titles)):
# 	ax.scatter(p1, p2, color='gray', s=15, zorder=10, marker='', edgecolors='k', linewidth=0.3)
# 	ax.text(p1, p2, title, color='black', fontdict={'size': 6})

# # Normal train author patches
# collected_patches = []
# for author in sorted(set(authors)):
# 	legend_patch = mpatches.Patch(color=colors_dict[author], label=legend_dict[author], zorder=2)
# 	collected_patches.append(legend_patch)

# Custom-made legend symbol patches for charters

# red_triangle = mlines.Line2D([], [], color='#E67365', marker='^', linestyle='None',
# 				markersize=5, label='Suger\'s charters')
# blue_triangle = mlines.Line2D([], [], color='#0DB4EF', marker='^', linestyle='None',
# 				markersize=5, label='Royal charters')
# yellow_triangle = mlines.Line2D([], [], color='#D8E665', marker='^', linestyle='None',
# 				markersize=5, label='Odo\'s charters')
# red_star = mlines.Line2D([], [], color='r', marker='*', linestyle='None',
# 				markersize=5, label='D Kar 286')
# gray_triangle = mlines.Line2D([], [], color='#ECECEC', marker='^', linestyle='None',
# 				markersize=5, label='Charter of Argentueil')

# collected_patches.append(red_triangle) 
# collected_patches.append(blue_triangle) 
# collected_patches.append(yellow_triangle) 
# collected_patches.append(red_star)
# collected_patches.append(gray_triangle)

# plt.legend(handles=collected_patches, fontsize=7, fancybox=True)

"""----------
plot loadings
"""
# ax2 = ax.twinx().twiny()
# l1, l2 = loadings[:,0], loadings[:,1]
# for x, y, l, font_size in zip(l1, l2, selected_features, font_dict_scores):
# 	if l in high_scorers:
# 		ax2.text(x, y, l, ha='center', va="center", color='k', fontdict={'size': font_size}, zorder=2)

# # Important to adjust margins first when function words fall outside plot
# # This is due to the axes aligning (def align).
# ax2.margins(x=1.5, y=1.5)

# align_xaxis(ax, 0, ax2, 0)
# align_yaxis(ax, 0, ax2, 0)

# plt.axhline(y=0, ls="--", lw=0.25, c='black', zorder=1)
# plt.axvline(x=0, ls="--", lw=0.25, c='black', zorder=1)

"""-----------------------------
plot layout and plotting command
"""
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
plt.tight_layout()
plt.show()
fig.savefig("/Users/jedgusse/compstyl/output/fig_output/heloise-abel-contours.pdf", bbox_inches='tight', transparent=True, format='pdf')
