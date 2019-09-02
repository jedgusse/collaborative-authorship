 #!/usr/bin/env

from binascii import hexlify
from collections import Counter
from itertools import combinations
from matplotlib import rcParams
from preprocess import DataReader, Vectorizer
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from string import punctuation
import argparse
import colorsys
import glob
import itertools
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import random
import seaborn.apionly as sns
import sys
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from tqdm import trange

def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""

	# Note that if base_cmap is a string or None, you can simply do
	#	return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:

	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)

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

def change_intensity(color, amount=0.5):
	"""
	Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.

	Examples:
	>> change_intensity('g', 0.3)
	>> change_intensity('#F034A3', 0.6)
	>> change_intensity((.3,.55,.1), 0.5)
	https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

	setting an amount < 1 lightens
	setting an amount > 1 darkens too

	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

class PrinCompAnal:

	""" |--- Principal Components Analysis ---|
		::: Plots PCA Plot ::: """

	def __init__(self, authors, titles, X, features, sample_size, n_components, show_pc2_pc3):
		self.authors = authors
		self.titles = titles
		self.X = X
		self.features = features
		self.sample_size = sample_size
		self.n_components = n_components
		self.show_pc2_pc3 = show_pc2_pc3

	def plot(self, show_samples, show_loadings, sbrn_plt):

		# Normalizer and Delta perform badly
		# They flatten out all difference in a PCA plot
		
		# Enables to see PC 2 and 3 in 2D
		if self.show_pc2_pc3 == True:
			pca = PCA(n_components=self.n_components + 1)
		else:
			pca = PCA(n_components=self.n_components)
		
		X_bar = pca.fit_transform(self.X)
		var_exp = pca.explained_variance_ratio_
		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		if self.n_components == 3:
			var_pc3 = np.round(var_exp[2]*100, decimals=2)
		if self.show_pc2_pc3 == True:
			var_pc3 = np.round(var_exp[2]*100, decimals=2)
		
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)
		comps = pca.components_
		comps = comps.transpose()
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(self.features, comps[:,0]), \
								  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(self.features, comps[:,1]), \
						   		  key=lambda tup: tup[1], reverse=True)


		print("Explained variance: ", explained_variance)
		print("Number of words: ", len(self.features))
		print("Sample size : ", self.sample_size)
		
		# Line that calls font (for doctoral thesis lay-out)
		# Custom fonts should be added to the matplotlibrc parameter file: cd '/Users/jedgusse/.matplotlib'.
		# Make sure the font is in the font "library" (not just font book!)
		# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib
		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['Alfios']

		if sbrn_plt == False:

			# Generate color dictionary
			color_dict = {author:index for index, author in enumerate(sorted(set(self.authors)))}
			cmap = discrete_cmap(len(color_dict), base_cmap='brg')

			legend_dictionary = {'Alanus-de-Insulis': 'Alan of Lille',
								 'Anselmus-Cantuariensis': 'Anselm of Canterbury',
								 'Anselmus-Laudunensis': 'Anselm of Laon',
								 'Bernard-Claraevallensis': 'Bernard of Clairvaux',
								 'Bruno-Carthusianorum': 'Bruno of Cologne',
								 'Gerhohus-Reicherspergensis': 'Gerhoch of Reichersberg',
								 'Gislebertus-Porretanus': 'Gilbert of Poitiers',
								 'Gualterus-de-Castellione': 'Walter of Châtillon',
								 'Guibertus-de-Novigento': 'Guibert of Nogent',
								 'Guib': 'Guibert of Gembloux',
								 'Guillelmus-de-S-Theodorico': 'William of Saint Thierry',
								 'Guillelmus-de-Conchis': 'William of Conches',
								 'Hildebertus-Lavardinensis': 'Hildebert de Lavardin',
								 'Honorius-Augustodunensis': 'Honorius Augustodunensis',
								 'Hugo-de-S-Victore': 'Hugo of Saint Victor',
								 'Ivo-Carnotensis': 'Ivo of Chartres',
								 'Joannes-Saresberiensis': 'John of Salisbury',
								 'Petrus-Abaelardus': 'Peter Abelard',
								 'Petrus-Cellensis': 'Peter of Celle',
								 'Petrus-Damianus': 'Peter Damian',
								 'Petrus-Lombardus': 'Peter Lombard',
								 'Petrus-Venerabilis': 'Peter the Venerable',
								 'vita-simonis': 'vita simonis',
								 'Rupertus-Tuitiensis': 'Rupert of Deutz',
								 'anon': 'Vita (autobiographical passages)',
								 'ysagoge': 'Ysagoge in theologiam',
								 'Hild': 'Hildegard of Bingen',
								 'Sugerius-sancti-Dionysii': 'Suger of Saint-Denis',
								 'Odo-de-Deogilo': 'Odo of Deuil',
								 'Wilhelmus-Sancti-Dionysii': 'William of Saint-Denis',
								 'Ekbert': 'Ekbert',
								 'Elisabeth': 'Unattributed',
								 'Petr-Abel': 'Peter Abelard (?)',
								 'Heloysa': 'Heloise (?)',
								 'mulier': '<M>ulier',
								 'vir': '<V>ir',
								 'tegernsee': 'Tegernsee',
								 'Theodericus-Epternacensis': 'Theoderic of Echternach',
								 'Vita-Hildegardis': 'Vita Hildegardis'}
			customized_colors = {'highlight': 'r', 
								 'anon': '#FD5960',
								'Alanus-de-Insulis': '#C0ED53',
								 'Anselmus-Cantuariensis': '#668DA6',
								 'Anselmus-Laudunensis': '#FCEEE4',
								 'Bern': '#668DA6',
								 'Bernardus-Claraevallensis': 'r',
								 'Bruno-Carthusianorum': '#40C3F1',
								 'Gerhohus-Reicherspergensis': '#C80C16',
								 'Gislebertus-Porretanus': '',
								 'Gualterus-de-Castellione': '#C80C16',
								 'Guibertus-de-Novigento': '#4BD4F2',
								 'Guillelmus-de-S-Theodorico': '#41AF4F',
								 'Guillelmus-de-Conchis': '#41AF4F',
								 'Hild': '#F8BA00',
								 'Hildebertus-Lavardinensis': '',
								 'Honorius-Augustodunensis': '#F3823C',
								 'Hugo-de-S-Victore': '#C5C9C9',
								 'Ivo-Carnotensis': '#D43F45',
								 'Joannes-Saresberiensis': '#C8CCD1',
								 'Sugerius-sancti-Dionysii': '#E67365',
								 'Odo-de-Deogilo': '#D8E665',
								 'Wilhelmus-Sancti-Dionysii': '#65E674',
								 'Petrus-Abaelardus': '#C41F24',
								 'Petrus-Cellensis': '#FEDF39',
								 'Petrus-Damianus': '',
								 'Petrus-Lombardus': '',
								 'Petrus-Venerabilis': '#00A2FF',
								 'vita-simonis': '#E31F4E',
								 # 'anon': 'g',
								 'ysagoge': '#FCEEE4',
								 'Intra-brevis': '#607AC4',
								 'Intra-perfectum': '#9CC2E0',
								 'Extra-pre-1140': '#ED3833',
								 'Extra-1140-1145': '#FC6D66',
								 'Extra-post-1145': '#F4AD8A',
								 'Nicolaus-Clareuallensis': '#4AB936',
								 'Nicolaus-Claraevallensis': '#4AB936',
								 'Elisabeth': '#53B466', 
								 'Ekbert': '#FDA25A',
								 'Petr-Abel': '#F77C6F',
								 'Heloysa': '#53B466',
								 'vir': '#F77C6F',
								 'mulier': '#53B466',
								 'tegernsee': '#BBA1FE',
								 'misc': 'r',
								 'Gaufridus-Autissiodorensis': 'r',
								 'Theodericus-Epternacensis': '#FDA660',
								 'Theoderic-dubium':'#757676',
								 'Godefridus-Sancti-Disibodi-dubium': '#EF5FA7',
								 'Vita-Hildegardis': '#757676',
								 'Guib': '#88BEDC',
								 'Guibertus-Gemblacensis': '#88BEDC',
								 'Hild': '#F8BA00',
								 'Other': '#757676',
								 'Sisters': '#00F3F3'
								 }

			if show_samples == True:

				fig = plt.figure(figsize=(4.7,3.2))
				# fig = plt.figure(figsize=(2.81, 1.91))
				# fig = plt.figure(figsize=(8,5))

				"""3D PROJECTION OF PCA CLUSTER PLOT"""
				if self.n_components == 3:
					ax = fig.add_subplot(111, projection='3d')
					x1, x2, x3 = X_bar[:,0], X_bar[:,1], X_bar[:,2]

					for index, (p1, p2, p3, a, title) in enumerate(zip(x1, x2, x3, self.authors, self.titles)):
						# Use line to generate colors. Do not forget to adjust legend.
						# ax.scatter(p1, p2, marker='o', color=cmap(color_dict[a]), s=20)

						# Use line to customize colors. Do not forget to adjust legend.
						if title.split('_')[0] in ['Ep-9', 'Ep-10', 'Ep-11', 'Ep-12', 'Ep-13', 'Ep-14', 'Epp-all']:
							markersymbol = '*'
							markersize = 25
						elif a in ['mulier', 'vir', 'Hild']:
							markersymbol = '^'
							markersize = 30
						elif title.split('_')[0] in ['institutiones-nostre', 'Ep-28']:
							markersymbol = 's'
							markersize = 20
						elif title.split('_')[0] in ['Epistulae-ad-Hildegardem', 'post-1173-epp']:
							markersymbol = '^'
							markersize = 20
						else:
							markersymbol = 'o'
							markersize = 20

						text = title.split('_')[0]
						sample_n = title.split('_')[1]

						# if text.split('-')[0] in ['Ep']:
							# text = text.split('-')[1] + '-' + sample_n
						if text == 'Epp-all':
							text = ''
						elif text == 'briefe':
							text = sample_n
						elif title.split('_')[0] == 'institutiones-nostre':
							text = 'IN'
						else:
							# text = sample_n
							text = ''


						# if title.split('_')[0] in ['auto-without-letters']:
						# 	ax.scatter(p1, p2, p3, marker='^', color='r', s=markersize, zorder=3000,
						# 		   edgecolors=change_intensity(customized_colors[a], 1.3), linewidth=0.3)
						# else:
						ax.scatter(p1, p2, p3, marker=markersymbol, color=customized_colors[a], s=markersize, zorder=3,
							   edgecolors=change_intensity(customized_colors[a], 1.3), linewidth=0.3)

						ax.text(p1-0.2, p2+0.2, p3+0.2, text, ha='center',
								va='center', color='k', fontdict={'size': 5}, zorder=1000)

					# Legend settings (code for making a legend)
					collected_patches = []
					# for author in sorted(set(self.authors)):
						# if author != 'Ekbert': 
						# 	author = 'anything-else'
						# legend_patch = mpatches.Patch(color=cmap(color_dict[author]), label=author.split('-')[0])
						# legend_patch = mpatches.Patch(color=color_dict[author], label=legend_dictionary[author])
						# collected_patches.append(legend_patch)
					# plt.legend(handles=collected_patches, fontsize=7, fancybox=True)

					ax.set_xlabel('PC 1: {}%'.format(var_pc1))
					ax.set_ylabel('PC 2: {}%'.format(var_pc2))
					ax.set_zlabel('PC 3: {}%'.format(var_pc3))

					plt.tight_layout()
					plt.show()

					fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", transparent=True, format='pdf')
				
				else:
					"""2D PROJECTION OF PCA CLUSTER PLOT"""
					ax = fig.add_subplot(111)
					if self.show_pc2_pc3 == True:
						x1, x2 = X_bar[:,1], X_bar[:,2]
					else:
						x1, x2 = X_bar[:,0], X_bar[:,1]

					#ax.scatter(x1, x2, x3, 100, zorder=5, edgecolors='none', facecolors='none', cmap='rainbow')
					for index, (p1, p2, a, title) in enumerate(zip(x1, x2, self.authors, self.titles)):
						# Use line to generate colors. Do not forget to adjust legend.
						# ax.scatter(p1, p2, marker='o', color=cmap(color_dict[a]), s=20)
						
						if title.split('_')[0] in ['Ep-9', 'Ep-10', 'Ep-11', 'Ep-12', 'Ep-13', 'Ep-14', 'Epp-all']:
							markersymbol = '*'
							markersize = 25
						elif a in ['mulier', 'vir']:
							markersymbol = '^'
							markersize = 40
						elif title.split('_')[0] == 'institutiones-nostre':
							markersymbol = 's'
							markersize = 20
						else:
							markersymbol = 'o'
							markersize = 20

						text = title.split('_')[0]
						text = ''

						ax.text(p1, p2, text, color='black', fontdict={'size': 6})

						ax.scatter(p1, p2, marker=markersymbol, color=customized_colors[a], s=markersize, zorder=3, 
									   edgecolors=change_intensity(customized_colors[a], 1.3), linewidth=0.3)
						
						if a == 'Theoderic-dubium':
							ax.text(p1-0.2, p2+0.2, title.split('_')[1], ha='center',
										va='center', color='k', fontdict={'size': 5})
						
					# collected_patches = []
					# for author in sorted(set(self.authors)):
					# 	# legend_patch = mpatches.Patch(color=cmap(color_dict[author]), label=author.split('-')[0])
					# 	legend_patch = mpatches.Patch(color=customized_colors[author], label=legend_dictionary[author])
					# 	collected_patches.append(legend_patch)
					# plt.legend(handles=collected_patches, fontsize=7, fancybox=True)

					# ax.set_xlabel('PC 1: {}%'.format(var_pc1))
					# ax.set_ylabel('PC 2: {}%'.format(var_pc2))

					ax2 = ax.twinx().twiny()
					l1, l2 = loadings[:,0], loadings[:,1]
					ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none', zorder=3)
					if show_loadings == True:
						rcParams['font.family'] = 'sans-serif'
						rcParams['font.sans-serif'] = ['Alfios']
						for x, y, l in zip(l1, l2, self.features):
							ax2.text(x, y, l, ha='center', va="center", color="black",
							fontdict={'size': 7}, zorder=4)

						# Align axes

						# Important to adjust margins first when function words fall outside plot
						# This is due to the axes aligning (def align).
						ax2.margins(x=0.15, y=0.15)

						align_xaxis(ax, 0, ax2, 0)
						align_yaxis(ax, 0, ax2, 0)

						if self.show_pc2_pc3 == False:
							ax.set_xlabel('PC 1: {}%'.format(var_pc1))
							ax.set_ylabel('PC 2: {}%'.format(var_pc2))
						elif self.show_pc2_pc3 == True:
							ax.set_xlabel('PC 2: {}%'.format(var_pc2))
							ax.set_ylabel('PC 3: {}%'.format(var_pc3))

						plt.axhline(y=0, ls="--", lw=0.25, c='black', zorder=1)
						plt.axvline(x=0, ls="--", lw=0.25, c='black', zorder=1)
						
						plt.tight_layout()
						plt.show()
					
					elif show_loadings == False:

						align_xaxis(ax, 0, ax2, 0)
						align_yaxis(ax, 0, ax2, 0)

						if self.show_pc2_pc3 == False:
							ax.set_xlabel('PC 1: {}%'.format(var_pc1))
							ax.set_ylabel('PC 2: {}%'.format(var_pc2))
						elif self.show_pc2_pc3 == True:
							ax.set_xlabel('PC 2: {}%'.format(var_pc2))
							ax.set_ylabel('PC 3: {}%'.format(var_pc3))

						plt.axhline(y=0, ls="--", lw=0.5, c='0.75', zorder=1)
						plt.axvline(x=0, ls="--", lw=0.5, c='0.75', zorder=1)

						plt.tight_layout()
						plt.show()

						# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

					fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", transparent=True, format='pdf')

			elif show_samples == False:

				fig = plt.figure(figsize=(8, 6))
				ax2 = fig.add_subplot(111)
				l1, l2 = loadings[:,0], loadings[:,1]
				ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none')
				for x, y, l in zip(l1, l2, self.features):
					ax2.text(x, y, l, ha='center', va='center', color='black',
						fontdict={'family': 'Arial', 'size': 6})

				ax2.set_xlabel('PC1')
				ax2.set_ylabel('PC2')

				align_xaxis(ax, 0, ax2, 0)
				align_yaxis(ax, 0, ax2, 0)

				plt.axhline(y=0, ls="--", lw=0.5, c='0.75', zorder=1)
				plt.axvline(x=0, ls="--", lw=0.5, c='0.75', zorder=1)

				plt.tight_layout()
				plt.show()
				fig.savefig("/Users/jedgusse/compstyl/output/fig_output/pcafig.pdf", bbox_inches='tight', transparent=True, format='pdf')

				# Converting PDF to PNG, use pdftoppm in terminal and -rx -ry for resolution settings

		else:

			data = [(title.split("_")[0], author, pc1, pc2) for [pc1, pc2], title, author in zip(X_bar, self.titles, self.authors)]
			df = pd.DataFrame(data, columns=['title', 'author', 'PC1', 'PC2'])

			# Get the x in an array
			sns.set_style('darkgrid')
			sns_plot = sns.lmplot('PC1', 'PC2', data=df, fit_reg=False, hue="author",
					   scatter_kws={"marker": "+","s": 100}, markers='o', legend=False)

			plt.legend(loc='upper right')
			plt.tight_layout()
			plt.show()

			sns_plot.savefig("/Users/jedgusse/compstyl/output/fig_output/pcasbrn.pdf")
			
class GephiNetworks:

	""" |--- Gephi Networks (k-NN Networks) ---|
		::: Yields k-Nearest Neighbor Network ::: """

	def __init__(self, folder_location, sample_size, invalid_words):
		self.folder_location = folder_location
		self.sample_size = sample_size
		self.invalid_words = invalid_words

	def plot(self, feat_range, random_sampling, corpus_size):

		# This is the standard number of neighbors. This cannot change unless the code changes.
		n_nbrs = 4

		# 3 neighbors for each sample is argued to make up enough consensus
		# Try to make a consensus of distance measures
		# Use cosine, euclidean and manhattan distance, and make consensus tree (inspired by Eder)
		# Also search over ranges of features to make the visualization less biased

		metric_dictionary = {'manhattan': 'manhattan', 'cosine': 'cosine', 'euclidean': 'euclidean'}

		rnd_dct = {'n_samples': 140,
				   'smooth_train': True, 
				   'smooth_test': False}

		authors, titles, texts = DataReader(self.folder_location, self.sample_size, {}, rnd_dct
										).metadata(sampling=True,
										type='folder',
										randomization=False)

		# random Stratified Sampling 
		# each sample receives its sampling fraction corresponding to proportionate number of samples

		corpus_size = corpus_size*1000

		if random_sampling == 'stratified':
			strata_proportions = {title.split('_')[0]: np.int(np.round(int(title.split('_')[-1]) / len(titles) * corpus_size / self.sample_size)) for title in titles}
			# print('::: corpus is being stratified to {} words in following proportions : '.format(str(corpus_size)))
			# print(strata_proportions, ' :::')
			strat_titles = []
			for stratum in strata_proportions:
				strata = [title for title in titles if stratum == title.split('_')[0]]
				sampling_fraction = strata_proportions[stratum]
				local_rand_strat_titles = random.sample(strata, sampling_fraction)
				strat_titles.append(local_rand_strat_titles)
			strat_titles = sum(strat_titles, [])
			strat_authors = [author for author, title in zip(authors, titles) if title in strat_titles]
			strat_texts = [text for title, text in zip(titles, texts) if title in strat_titles]
			
			titles = strat_titles
			authors = strat_authors
			texts = strat_texts

		if random_sampling == 'simple':
			all_titles = {title.split('_')[0] for title in titles}
			n_samples = np.int(np.floor(corpus_size / self.sample_size / len(all_titles)))
			strata_proportions = {title: n_samples for title in all_titles}
			strat_titles = []
			for stratum in strata_proportions:
				strata = [title for title in titles if stratum == title.split('_')[0]]
				sampling_fraction = strata_proportions[stratum]
				local_rand_strat_titles = random.sample(strata, sampling_fraction)
				strat_titles.append(local_rand_strat_titles)
			strat_titles = sum(strat_titles, [])
			strat_authors = [author for author, title in zip(authors, titles) if title in strat_titles]
			strat_texts = [text for title, text in zip(titles, texts) if title in strat_titles]

			titles = strat_titles
			authors = strat_authors
			texts = strat_texts

		fob_nodes = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_nodes.csv", "w")
		fob_edges = open("/Users/jedgusse/compstyl/output/gephi_output/gephi_edges.csv", "w")

		fob_nodes.write("Id" + "\t" + "Work" + "\t" + "Author" + "\n")
		fob_edges.write("Source" + "\t" + "Target" + "\t" + "Type" + "\t" + "Weight" + "\n")

		# Build up consensus distances of different feature ranges and different metrics
		exhsearch_data = []
		function_words_only = open('/Users/jedgusse/compstyl/params/fword_list.txt').read().split()
		
		_, possible_feats, _ = Vectorizer(texts, self.invalid_words,
										  n_feats=feat_range[-1],
										  feat_scaling='standard_scaler',
										  analyzer='word',
										  vocab=function_words_only
										  ).tfidf(smoothing=True)

		for n_feats in tqdm(feat_range, postfix='n feats'):
			# print("::: running through feature range {} ::: ".format(str(n_feats)))
			tfidf_vectors, tfidf_features, scaling_model = Vectorizer(texts, self.invalid_words,
																	  n_feats=n_feats,
																	  feat_scaling='standard_scaler',
																	  analyzer='word',
																	  vocab=possible_feats[:n_feats]
																	  ).tfidf(smoothing=True)

			if n_feats == feat_range[-1]:
				pass
				# print("FEATURES: ", ", ".join(tfidf_features))
			for metric in tqdm(metric_dictionary, postfix='metric'):
				model = NearestNeighbors(n_neighbors=n_nbrs,
										algorithm='brute',
										metric=metric_dictionary[metric],
										).fit(tfidf_vectors)
				distances, indices = model.kneighbors(tfidf_vectors)
				
				# Distances are normalized in order for valid ground for comparison
				all_distances = []
				for distance_vector in distances:
					for value in distance_vector:
						if value != 0.0:
							all_distances.append(value)

				all_distances = np.array(all_distances)
				highest_value = all_distances[np.argmin(all_distances)]
				lowest_value = all_distances[np.argmax(all_distances)]
				normalized_distances = (distances - lowest_value) / (highest_value - lowest_value)
				
				# Distances appended to dataframe
				for distance_vec, index_vec in zip(normalized_distances, indices):
					data_tup = ('{} feats, {}'.format(str(n_feats), metric_dictionary[metric]),
								titles[index_vec[0]], 
								titles[index_vec[1]], distance_vec[1],
								titles[index_vec[2]], distance_vec[2],
								titles[index_vec[3]], distance_vec[3])
					exhsearch_data.append(data_tup)

		# Entire collected dataframe
		df = pd.DataFrame(exhsearch_data, columns=['exp', 'node', 'neighbor 1', 'dst 1', 'neighbor 2', 
										 'dst 2', 'neighbor 3', 'dst 3']).sort_values(by='node', ascending=0)
		final_data = []
		weights= []
		node_orientation  = {title: idx+1 for idx, title in enumerate(titles)}
		for idx, (author, title) in enumerate(zip(authors, titles)):
			neighbors = []
			dsts = []
			# Pool all neighbors and distances together (ignore ranking of nb1, nb2, etc.)
			for num in range(1, n_nbrs):
				neighbors.append([neighb for neighb in df[df['node']==title]['neighbor {}'.format(str(num))]])
				dsts.append([neighb for neighb in df[df['node']==title]['dst {}'.format(str(num))]])
			neighbors = sum(neighbors, [])
			dsts = sum(dsts, [])

			model = CountVectorizer(lowercase=False, token_pattern=r"[^=]*")
			count_dict = model.fit_transform(neighbors)
			names = [i for i in model.get_feature_names() if i != '']
			
			# Collect all the candidates per sample that were chosen by the algorithm as nearest neighbor at least once
			candidate_dict = {neighbor: [] for neighbor in names}
			for nbr, dst in zip(neighbors, dsts):
				candidate_dict[nbr].append(dst)
			candidate_dict = {nbr: np.mean(candidate_dict[nbr])*len(candidate_dict[nbr]) for nbr in candidate_dict}
			candidate_dict = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)

			fob_nodes.write(str(idx + 1) + "\t" + str(title.split('_')[0]) + "\t" + str(author) + "\n")
			data_tup = (title,)
			for candtitle, weight in candidate_dict[:8]:
				data_tup = data_tup + (candtitle, weight,)
				weights.append(weight)
				fob_edges.write(str(idx+1) + "\t" + str(node_orientation[candtitle]) + "\t" + "Undirected" + "\t" + str(weight) + "\n")
			final_data.append(data_tup)

		# Prepare column names for dataframe
		longest = np.int((len(final_data[np.argmax([len(i) for i in final_data])]) - 1) / 2)
		columns = sum([['neighbor {}'.format(str(i)), 'dst {}'.format(str(i))] for i in range(1, longest+1)], [])
		columns.insert(0, 'node')
		final_df = pd.DataFrame(final_data, columns=columns).sort_values(by='node', ascending=0)
		print(final_df)

		# Results
		# print('::: RESULTS :::')
		# print(final_df.head())
		# print('::: VARIANCE BETWEEN DISTANCES :::')
		return np.var(np.array(weights))

class VoronoiDiagram:

	""" |--- Gives realistic estimate of 2D-Decision boundary ---|
		::: Only takes grid_doc_vectors ::: """

	def __init__(self):
		self.grid_doc_vectors = grid_doc_vectors
		self.Y_train = Y_train
		self.y_predicted = y_predicted
		self.best_n_feats = best_n_feats
		self.ordered_authors = ordered_authors
		self.ordered_titles = ordered_titles

	def plot(self):
		
		colours = {'Bern': '#000000', 'NicCl': '#000000', 'lec': '#ff0000', 
				   'ro': '#ff0000', 'Alain': '#000000', 'AnsLaon': '#000000', 
			   	   'EberYpr': '#000000', 'Geof': '#000000', 'GilPoit': '#000000'}

		# Plotting SVM

		# Dimensions of the data reduced in 2 steps - from 300 to 50, then from 50 to 2 (this is a strong recommendation).
		# t-SNE (t-Distributed Stochastic Neighbor Embedding): t-SNE is a tool for data visualization. 
		# Local similarities are preserved by this embedding. 
		# t-SNE converts distances between data in the original space to probabilities.
		# In contrast to, e.g., PCA, t-SNE has a non-convex objective function. The objective function is minimized using a gradient descent 
		# optimization that is initiated randomly. As a result, it is possible that different runs give you different solutions.

		# First, reach back to original values, and put in the new y predictions in order to draw up the Voronoi diagram, which is basically
		# a 1-Nearest Neighbour fitting. (For the Voronoi representation, see MLA	
		# Migut, M. A., Marcel Worring, and Cor J. Veenman. "Visualizing multi-dimensional decision boundaries in 2D." 
		# Data Mining and Knowledge Discovery 29.1 (2015): 273-295.)

		# IMPORTANT: we take the grid_doc_vectors (not original data), those feature vectors which the SVM itself has made the decision on.
		# We extend the y vector with the predicted material

		print('::: running t-SNE for dimensionality reduction :::')

		y = np.append(self.Y_train, self.y_predicted, axis=0)

		# If features still too many, truncate the grid_doc_vectors to reasonable amount, then optimize further
		# A larger / denser dataset requires a larger perplexity

		if self.best_n_feats < 50:
			X_embedded = TSNE(n_components=2, perplexity=40, verbose=2 \
			).fit_transform(self.grid_doc_vectors)
		else:
			X_reduced = TruncatedSVD(n_components=50, random_state=0 \
			).fit_transform(self.grid_doc_vectors)
			X_embedded = TSNE(n_components=2, perplexity=40, verbose=2 \
			).fit_transform(X_reduced)

		# create meshgrid
		resolution = 100 # 100x100 background pixels
		X2d_xmin, X2d_xmax = np.min(X_embedded[:,0]), np.max(X_embedded[:,0])
		X2d_ymin, X2d_ymax = np.min(X_embedded[:,1]), np.max(X_embedded[:,1])
		xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), \
							 np.linspace(X2d_ymin, X2d_ymax, resolution))

		# Approximate Voronoi tesselation on resolution x resolution grid using 1-NN
		background_model = KNeighborsClassifier(n_neighbors=1).fit(X_embedded, y)
		voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
		voronoiBackground = voronoiBackground.reshape((resolution, resolution))

		# (http://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data)
		vor = Voronoi(X_embedded)

		fig = plt.figure(figsize=(10,8))

		# Define colour mapping

		plt.contourf(xx, yy, voronoiBackground, levels=[0, 0.5, 1], colors=('#eaeaea', '#b4b4b4'))
		ax = fig.add_subplot(111)
		
		ax.scatter(X_embedded[:,0], X_embedded[:,1], 100, edgecolors='none', facecolors='none')
		for p1, p2, a, title in zip(X_embedded[:,0], X_embedded[:,1], self.ordered_authors, self.ordered_titles):
			ax.text(p1, p2, title[:2] + '_' + title.split("_")[1], ha='center',
			va='center', color=colours[a], fontdict={'size': 7})
		for vpair in vor.ridge_vertices:
			if vpair[0] >= 0 and vpair[1] >= 0:
				v0 = vor.vertices[vpair[0]]
				v1 = vor.vertices[vpair[1]]
				# Draw a line from v0 to v1.
				plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=0.3, linestyle='--')

		ax.set_xlabel('F1 (Conditional probability)')
		ax.set_ylabel('F2 (Conditional probability)')

		plt.axis([X2d_xmin, X2d_xmax, X2d_ymin, X2d_ymax])

		plt.show()

		fig.savefig("/Users/jedgusse/stylofactory/output/fig_output/voronoi_fig.pdf", \
		transparent=True, format='pdf')

class RollingDelta:

	""" |--- Rolling Delta ---|
		::: Roll test vectors over centroid train vector ::: """

	def __init__(self, folder_location, n_feats, invalid_words, sample_size, step_size, test_dict, rnd_dct):
		self.folder_location = folder_location
		self.n_feats = n_feats
		self.invalid_words = invalid_words
		self.sample_size = sample_size
		self.step_size = step_size
		self.test_dict = test_dict
		self.rnd_dct = rnd_dct

	def plot(self):

		# Make a train_test_split
		# The training corpus is the benchmark corpus

		train_data = []
		train_metadata = []
		
		test_data = []
		test_metadata = []

		# Make a split by using the predefined test_dictionary

		print("::: test - train - split :::")

		for filename in glob.glob(self.folder_location + '/*'):
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]

			if title not in self.test_dict.values():
				author, title, text = DataReader(filename, 
										self.sample_size, self.test_dict,
										self.rnd_dct).metadata(sampling=True,
										type='file', randomization=False)
				train_metadata.append((author, title))
				train_data.append(text)

			elif title in self.test_dict.values():
				author, title, text = DataReader(filename, 
										self.sample_size, self.test_dict,
										self.rnd_dct).metadata(sampling=False, 
										type='file', randomization=False)
				test_metadata.append((author, title))
				test_data.append(text.split())

		# Unnest nested list
		# Preparing the two corpora for take-off
		train_data = sum(train_data, [])

		print("::: vectorizing training corpus :::")

		# Vectorize training data
		doc_vectors, features = Vectorizer(train_data, self.invalid_words,
									  n_feats=self.n_feats,
									  feat_scaling=False,
									  analyzer='word',
									  vocab=None
									  ).raw()

		# We first turn our raw counts into relative frequencies
		relative_vectors = [vector / np.sum(vector) for vector in doc_vectors]
		
		# We produce a standard deviation vector, that will later serve to give more weight to 
		# highly changeable words and serves to
		# boost words that have a low frequency. This is a normal Delta procedure.
		# We only calculate the standard deviation on the benchmark corpus, 
		# since that is the distribution against which we want to compare
		stdev_vector = np.std(relative_vectors, axis = 0)

		# We make a centroid vector for the benchmark corpus
		centroid_vector = np.mean(relative_vectors, axis=0)

		# Now we have turned all the raw counts of the benchmark corpus into relative 
		# frequencies, and there is a centroid vector
		# which counts as a standard against which the test corpus can be compared.

		# We now divide the individual test texts in the given sample lengths, 
		# taking into account the step_size of overlap
		# This is the "shingling" procedure, where we get overlap, where we get windows	

		# Get highest x value
		lengths = np.array([len(text) for text in test_data])
		maxx = lengths[np.argmax(lengths)]

		print("::: making step-sized windows and rolling out test data :::")

		all_data = []
		for (author, title), test_text in zip(test_metadata, test_data):

			steps = np.arange(0, len(test_text), self.step_size)
			step_ranges = []

			windowed_samples = []
			for each_begin in steps:
				sample_range = range(each_begin, each_begin + self.sample_size)
				step_ranges.append(sample_range)
				text_sample = []
				for index, word in enumerate(test_text):
					if index in sample_range:
						text_sample.append(word)
				windowed_samples.append(text_sample)

			# Now we change the samples to numerical values, using the features as determined in code above
			# Only allow text samples that have desired sample length

			window_vectors = []
			for text_sample in windowed_samples:
				if len(text_sample) == self.sample_size:
					vector = []
					counter = Counter(text_sample)
					for feature in features:
						vector.append(counter[feature])
					window_vectors.append(vector)
			window_vectors = np.asarray(window_vectors)

			window_relative = [vector / np.sum(vector) for vector in window_vectors]

			delta_scores = []
			for vector in window_relative:
				delta_distances = np.mean(np.absolute(centroid_vector - vector) / stdev_vector)
				delta_score = np.mean(delta_distances)
				delta_scores.append(delta_score)

			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples) \
			if len(sample) == self.sample_size]

			data = [(author, title, x+1, y) for x, y in zip(x_values, delta_scores)]
			all_data.append(data)

		all_data = sum(all_data, [])
		df = pd.DataFrame(all_data, columns=['author', 'title', 'x-value', 'delta-value'])

		# Plot with seaborn

		fig = plt.figure(figsize=(20,5))

		sns.plt.title('Rolling Delta')
		sns.set(font_scale=0.5)
		sns.set_style("whitegrid")

		ax = sns.pointplot(x=df['x-value'], y=df['delta-value'], data=df, ci=5, scale=0.4, hue='author')

		ax.set_xlabel("Step Size: {} words, Sample Size: {} words".format(str(self.step_size), \
		str(self.sample_size)))
		ax.set_ylabel("Delta Score vs. Centroid Vector")

		# Set good x tick labels
		for ind, label in enumerate(ax.get_xticklabels()):
			if ind % 30 == 0:
				label.set_visible(True)
			else:
				label.set_visible(False)

		sns.plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/rollingdelta.pdf", bbox_inches='tight')

class IntrinsicPlagiarism:

	""" |--- Intrinsic Plagiarism ---|
		::: N-gram profiles and a sliding window with no reference corpus ::: """

	def __init__(self, folder_location, n_feats, invalid_words, sample_size, step_size):
		self.folder_location = folder_location
		self.n_feats = n_feats
		self.invalid_words = invalid_words
		self.sample_size = sample_size
		self.step_size = step_size

	def plot(self, support_ngrams, support_punct):
		
		# Make sure the analyzer is on the character or word level.
		analyzer = ''
		n = 3
		ngram_range = None
		if support_ngrams == False:
			analyzer += 'word'
			ngram_range = ((1,1))
		else:
			analyzer += 'char'
			# Advised by Stamatatos: the 3gram range
			ngram_range = ((n,n))

		# Open file and set up stepsized samples
		filename = glob.glob(self.folder_location+"/*")
		if len(filename) > 1:
			sys.exit("-- | ERROR: Intrinsic plagiarism detection can handle only 1 file")
		fob = open(filename[0])
		text = fob.read()
		bulk = []
		if support_punct == False:
			for feat in text.strip().split():
				feat = "".join([char for char in feat if char not in punctuation])
				bulk.append(feat)

		if analyzer == 'word':
			text = bulk
		elif analyzer == 'char':
			text = " ".join(bulk)

		# Make sure the texts are split when words are analyzed
		# Also the reconnection of features in the texts happens differently
		print("::: creating sliding windows :::")
		steps = np.arange(0, len(text), self.step_size)
		step_ranges = []
		windowed_samples = []
		for each_begin in steps:
			sample_range = range(each_begin, each_begin + self.sample_size)
			step_ranges.append(sample_range)
			text_sample = []
			for index, feat in enumerate(text):
				if index in sample_range:
					text_sample.append(feat)
			if len(text_sample) == self.sample_size:
				if analyzer == 'char':
					windowed_samples.append("".join(text_sample))
				elif analyzer == 'word':
					windowed_samples.append(" ".join(text_sample))
		if analyzer == 'word':
			text = " ".join(text)

		print("::: converting windows to document vectors :::")
		model = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, 
								stop_words=self.invalid_words)
		doc_vector = model.fit_transform([text]).toarray().flatten()
		doc_vector = doc_vector / len(text)
		vocab = model.get_feature_names()

		print("::: calculating dissimilarity measures :::")
		# Count with predefined vocabulary based on entire document
		ds = []
		model = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, 
								vocabulary=vocab, stop_words=self.invalid_words)
		for sample in windowed_samples:
			sample_vector = model.fit_transform([sample]).toarray().flatten()
			sample_vector = sample_vector / len(sample)
			dissimilarity_measure = np.power(np.mean(np.divide(2*(sample_vector - doc_vector), \
			sample_vector + doc_vector)), 2)
			ds.append(dissimilarity_measure)
		
		# Set up threshold; in calculating the threshold, ignore likely-to-be plagiarized passages 
		filter = np.mean(ds) + np.std(ds)
		averaged_ds = [i for i in ds if i <= filter]
		filter_threshold = np.mean(averaged_ds) + 2*np.std(averaged_ds)

		print("::: visualizing style change function «sc» :::")
		if analyzer == 'char':
			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples) \
			if len(sample) == self.sample_size]
		elif analyzer == 'word':
			x_values = [graphthing[-1] for graphthing, sample in zip(step_ranges, windowed_samples)]
		data = [(x+1, y, filter_threshold) for x, y in zip(x_values, ds)]
		df = pd.DataFrame(data, columns=['range', 'dissimilarity measure', 'filter_threshold'])

		# Exporting plagiarized text to database
		print("::: ranges and detected stylistic outliers :::")
		df_plag = []
		for s_range, dissimilarity_measure in zip(step_ranges, ds):
			if dissimilarity_measure >= filter_threshold:
				range_string = str(s_range[0]) + "-" + str(s_range[-1])
				plag_text = "".join(text[index] for index in s_range)
				df_plag.append((range_string, plag_text))
		df_plag = pd.DataFrame(df_plag, columns=['{} range'.format(analyzer), 'plagiarized'])
		print(df_plag)

		# Plot with seaborn

		fig = plt.figure(figsize=(20,5))

		sns.plt.title(r'Intrinsic plagiarism, ${}-profiling$'.format(analyzer))
		sns.set(font_scale=0.5)
		sns.set_style("darkgrid")

		ax = sns.pointplot(x=df['range'], y=df['dissimilarity measure'], data=df, ci=5, scale=0.4)
		ax.set_xlabel(r"Step Size: ${}$ {}s, Sample Size: ${}$ {}s".format(str(self.step_size), \
		analyzer, str(self.sample_size), analyzer))
		ax.set_ylabel(r"Dissimilarity measure ($d$)")

		# Plot red line
		plt.plot([0, step_ranges[-1][-1]], [filter_threshold, filter_threshold], '--', lw=0.75, color='r')

		# Set right xtick labels
		increment = np.int(np.round(len(ax.get_xticklabels())/10))
		visible_labels = range(0, len(ax.get_xticklabels()), increment)
		for idx, label in enumerate(ax.get_xticklabels()):
			if idx in visible_labels:
				label.set_visible(True)
			else:
				label.set_visible(False)

		sns.plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/intrinsic_plagiarism.pdf", bbox_inches='tight')

class HeatMap:

	def __init__(self, doc_vectors, features, authors, titles):
		self.doc_vectors = doc_vectors
		self.authors = authors
		self.titles = titles
		self.features = features 

	def plot(self):
		fig, ax = plt.subplots()
		
		distance_matrix = squareform(pdist(self.doc_vectors, 'cityblock'))
		heatmap = ax.pcolor(distance_matrix, cmap=plt.cm.Reds)
		
		ax.set_xticks(np.arange(distance_matrix.shape[0])+0.5, minor=False)
		ax.set_yticks(np.arange(distance_matrix.shape[1])+0.5, minor=False)
		
		ax.set_xticklabels(self.titles, minor=False, rotation=90)
		ax.set_yticklabels(self.titles, minor=False)
		
		plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/heatmap.pdf", bbox_inches='tight')

class LexicalRichness:

	def __init__(self, authors, titles, texts):
		self.authors = authors
		self.titles = titles
		self.texts = texts

	def plot(self, sample_size):

		# Place works in custom, preferred order in visualization
		works_order = ['sciu', 'lvm', 'ldo']
		works_order = ['matthaeus', 'marcus', 'lucas', 'ioannes']
		seq_authors = []
		seq_titles = []
		seq_texts = []
		for the_title in works_order:
			for author, title, text in zip(self.authors, self.titles, self.texts):
				if title.split('_')[0] == the_title:
					seq_authors.append(author)
					seq_titles.append(title)
					seq_texts.append(text)
		self.authors = seq_authors
		self.titles = seq_titles
		self.texts = seq_texts

		ranges_dict = {title.split('_')[0]: 0 for title in self.titles}
		for title in self.titles:
			if title.split('_')[0] in ranges_dict:
				ranges_dict[title.split('_')[0]] += 1
		for title in ranges_dict:
			ranges = []
			sample_idxs = list(range(0, ranges_dict[title]))
			for i in sample_idxs:
				ranges.append(i * sample_size + sample_size)
			ranges_dict[title] = ranges

		# Make custom indices so that the lines can be plotted next to each other
		ranges_dict_new = sum([each for each in ranges_dict.values()], [])
		custom_idcs = []
		for i in range(1, len(ranges_dict_new)+1):
			idx = i * sample_size
			custom_idcs.append(idx)

		data = []
		tt_ratio_dict = {title: [] for title in ranges_dict}
		for umb_title in ranges_dict:
			for author, title, text_sample, custom_idx in \
				zip(self.authors, self.titles, self.texts, custom_idcs):
				if title.split('_')[0] == umb_title:
					idx = int(title.split('_')[1])-1
					counter = Counter(text_sample.split())
					tt_ratio = len(counter)/sample_size
					tt_ratio_dict[umb_title].append(tt_ratio)
					data.append((ranges_dict[umb_title][idx], custom_idx, author, title, umb_title, tt_ratio))

		# colour_dict = {'sciu': '#D8E665', 
		# 			   'lvm': '#6597e6', 
		# 			   'ldo': '#e67465'}

		colour_dict = {'marcus': '#FFC857', 
					   'lucas': '#F4B9B2', 
					   'matthaeus': '#DE6B48',
					   'ioannes': '#7DBBC3'}
		
		df = pd.DataFrame(data, columns=['idx', 'custom_idx', 'author', 'title', 'Name of Work', 'TTR'])
		print(df)

		# Plot with seaborn

		fig = plt.figure(figsize=(8,3.5))

		sns.set(font_scale=0.7)
		sns.set_style('white')
		sns.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 3})

		ax = sns.pointplot(x=df['custom_idx'], y=df['TTR'], data=df, ci=5, scale=0.7, \
		hue='Name of Work', palette=colour_dict)
		ax.set_xlabel(r"$Word\ Index$", fontdict={'size': 7})
		ax.set_ylabel(r"$Type/Token\ Ratio$ (TTR)", fontdict={'size': 7})

		sns.despine(ax=ax)

		print("::: average type/token ratio :::")
		for title in tt_ratio_dict:
			ratios = np.array(tt_ratio_dict[title])
			mean = np.mean(ratios)
			plt.axhline(y=mean, linewidth=1.0, color=colour_dict[title], linestyle='--')
			print(mean, "\t", title)

		# Set good x tick labels
		labels = [item.get_text() for item in ax.get_xticklabels()]
		new_labels = [idx for label, idx in zip(labels, df['idx'])]
		ax.set_xticklabels(new_labels)

		for ind, label in enumerate(ax.get_xticklabels()):
			if ind % 4 == 0:
				label.set_visible(True)
			else:
				label.set_visible(False)

		sns.plt.show()
		fig.savefig("/Users/jedgusse/compstyl/output/fig_output/lexicalrichness.pdf", bbox_inches='tight')
