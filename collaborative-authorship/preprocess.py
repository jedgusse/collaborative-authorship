#!/usr/bin/env
# -*- coding: utf-8 -*-

from cltk.tokenize.sentence import TokenizeSentence
from collections import namedtuple as nt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (Normalizer,
                                   StandardScaler,
                                   FunctionTransformer)
from string import punctuation
import glob
import itertools
import numpy as np
import random
import re
import sys
import threading
import time

def windower(train_path, step_size, sample_len, rnd_dict):
	"""
	Function to split document into partially overlapping window samples
	"""

	data = {}
	for each_file_path in glob.glob(train_path + '/*'):
		author = each_file_path.split('/')[-1].split('_')[0]
		data[author] = []

	for each_file_path in glob.glob(train_path + '/*'):
		original_txt = open(each_file_path).read()
		clean_txt = re.sub('[%s]' % re.escape(punctuation), '', original_txt) # Escape punctuation and make characters lowercase
		clean_txt = re.sub('\d+', '', clean_txt)
		clean_txt = clean_txt.lower().split()	

		title = each_file_path.split('/')[-1].split('_')[1]		

		steps = np.arange(0, len(clean_txt), step_size)

		step_ranges = []
		for each_begin in steps:
			sample_range = range(each_begin, each_begin + sample_len)
			step_ranges.append(sample_range)
			text_sample = []
			for index, word in enumerate(clean_txt):
				if index in sample_range:
					text_sample.append(word)
			new_title = '{}-{}-{}'.format(str(title), str(each_begin), str(each_begin + sample_len))
			if len(text_sample) == sample_len:
				value = (new_title, ' '.join(text_sample))
				data[author].append(value)

		authors = []
		titles = []
		texts = []
		for key, lis in data.items():
			for tup in lis:
				authors.append(key)
				titles.append(tup[0])
				texts.append(tup[1])

		return authors, titles, texts

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

def deltavectorizer(X):
	    # "An expression of pure difference is what we need"
	    #  Burrows' Delta -> Absolute Z-scores
	    X = np.abs(stats.zscore(X))
	    X = np.nan_to_num(X)
	    return X

def randomizer(authors, titles, texts, sample_size, 
			   test_dict, n_samples, smooth_test):

	""" 
	Function for making random samples from texts.
	Random samples are composed by combining randomly selected sentences.
	"""

	sampled_authors = []
	sampled_titles = []
	sampled_texts = []

	# Make train-test dict
	# Texts under the same author name are collected in one pool and then randomized
	pooled_dict = {author: [] for author in authors}
	for author, title, text in zip(authors, titles, texts):
		if author in pooled_dict:
			pooled_dict[author].append((title, text))

	# Instantiate cltk Tokenizer
	tokenizer = TokenizeSentence('latin')

	for author in pooled_dict:
		# Pool together texts by same author
		pooled_titles = [tup[0] for tup in pooled_dict[author]]
		pooled_texts = [tup[1] for tup in pooled_dict[author]]

		if author in test_dict and test_dict[author] in pooled_titles and smooth_test == False:
			print("::: test set «{} {}» is sampled in ordinary slices :::".format(author, "+".join(pooled_titles)))
			bulk = []
			for ord_text in pooled_texts:
				for word in ord_text.strip().split():
					word = word.lower()
					word = "".join([char for char in word if char not in punctuation])
					word = word.lower()
					bulk.append(word)
				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]
				bulk = [bulk[i:i+sample_size] for i in range(0, len(bulk), sample_size)]
				for index, sample in enumerate(bulk):
					if len(sample) == sample_size: 
						sampled_authors.append(author)
						sampled_titles.append(test_dict[author] + "_{}".format(str(index + 1)))
						sampled_texts.append(" ".join(sample))

		else:
			# Make short random samples and add to sampled texts
			# Remove punctuation in the meantime
			print("::: training set «{} {}» is randomly sampled from corpus :::".format(author, \
				  "+".join(pooled_titles)))
			pooled_texts = " ".join(pooled_texts)
			pooled_texts = tokenizer.tokenize_sentences(pooled_texts)
			if len(pooled_texts) < 20:
				print("-----| ERROR: please check if input texts have punctuation, \
					   tokenization returned only {} sentence(s) |-----".format(len(pooled_texts)))
				break
			for _ in range(1, n_samples+1):
				random_sample = []
				while len(" ".join(random_sample).split()) <= sample_size:
					random_sample.append(random.choice(pooled_texts))
				for index, word in enumerate(random_sample):
					random_sample[index] = "".join([char for char in word if char not in punctuation])
				random_sample = " ".join(random_sample).split()[:sample_size]
				sampled_authors.append(author)
				sampled_titles.append('{}_{}'.format(pooled_titles[0], _))
				sampled_texts.append(" ".join(random_sample))

	return sampled_authors, sampled_titles, sampled_texts

def enclitic_split(input_str):
	# Feed string, returns lowercased text with split enclitic -que
	que_list = open("/Users/jedgusse/compstyl/params/que_list.txt").read().split()
	spaced_text = []
	for word in input_str.split():
		if word[-3:] == 'que' and word not in que_list:
			word = word.replace('que','') + ' que'
		spaced_text.append(word)
	spaced_text = " ".join(spaced_text)
	return spaced_text

def words_and_bigrams(text):
	words = re.findall(r'\w{1,}', text)
	for w in words:
		if w not in stop_words:
			yield w.lower()
		for i in range(len(words) - 2):
			if ' '.join(words[i:i+2]) not in stop_words:
				yield ' '.join(words[i:i+2]).lower()

class DataReader:

	""" |--- Defines metadata ---|
		::: Authors, Titles, Texts ::: """

	def __init__(self, folder_location, sample_size, test_dict, rnd_dict):
		self.folder_location = folder_location
		self.sample_size = sample_size
		self.test_dict = test_dict
		self.rnd_dict = rnd_dict

	def metadata(self, sampling, type, randomization):

		authors = []
		titles = []
		texts = []

		""" |--- Accepts both entire folders as files 
		::: More flexibility ---|"""

		if type == 'folder':
		
			for filename in glob.glob(self.folder_location + "/*"):
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split('_')[-1].split('.')[0]
								
				text = open(filename).read()
				if randomization == False:
					text = re.sub('[%s]' % re.escape(punctuation), '', text)
					text = re.sub('\d+', '', text)
					text = enclitic_split(text)
				# For randomization, punctuation is required so as to demarcate sentence constituents
				elif randomization == True:
					text = re.sub('\d+', '', text)

				bulk = text.lower().split()
				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]

				if sampling == True:
					if randomization == True:
						print("-- | ERROR: randomization and sampling both set to True")
						break
					else:
						bulk = [bulk[i:i+self.sample_size] for i \
								in range(0, len(bulk), self.sample_size)]
						for index, sample in enumerate(bulk):
							if len(sample) == self.sample_size:
								authors.append(author)
								titles.append(title + "_{}".format(str(index + 1)))
								texts.append(" ".join(sample))

				elif sampling == False:
					authors.append(author)
					titles.append(title)
					bulk = " ".join(bulk)
					texts.append(bulk)

			if randomization == True:
				authors, titles, texts = randomizer(authors, titles, texts,
									   	 self.sample_size, self.test_dict, 
									   	 n_samples=self.rnd_dict['n_samples'],
									   	 smooth_test=self.rnd_dict['smooth_test'])

			return authors, titles, texts

		elif type == 'file':

			# The input is not a folder location, but a filename
			# So change the variable name from here on out

			filename = self.folder_location

			bulk = []

			fob = open(filename)
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]
			text = fob.read()
			for word in text.strip().split():
				word = [char for char in word if char not in punctuation]
				word = "".join(word)
				word = word.lower()
				bulk.append(word)
			# Safety measure against empty strings in samples
			bulk = [word for word in bulk if word != ""]

			if sampling == True:
				bulk = [bulk[i:i+self.sample_size] for i in range(0, len(bulk), self.sample_size)]

				for index, sample in enumerate(bulk):
					if len(sample) == self.sample_size:
						authors.append(author)
						titles.append(title + "_{}".format(str(index + 1)))
						texts.append(" ".join(sample))

			else:
				texts = " ".join(bulk)

			return author, title, texts

class Vectorizer:

	""" |--- From flat text to document vectors ---|
		::: Document Vectors, Most Common Features ::: """

	def __init__(self, texts, stop_words, n_feats, feat_scaling, analyzer, vocab, ngram):
		self.texts = texts
		self.stop_words = stop_words
		self.n_feats = n_feats
		self.feat_scaling = feat_scaling
		self.analyzer = analyzer
		self.vocab = vocab
		self.ngram = ngram
		self.norm_dict = {'delta': FunctionTransformer(deltavectorizer), 
				   	   	  'normalizer': Normalizer(),
						  'standard_scaler': StandardScaler()}

	# Raw Vectorization

	def raw(self):

		# Text vectorization; array reversed to order of highest frequency
		# Vectorizer takes a list of strings

		# Define fed-in analyzer
		ngram_range = ((self.ngram,self.ngram))

		"""option where only words from vocab are taken into account"""
		model = CountVectorizer(stop_words=self.stop_words, 
							    max_features=self.n_feats,
							    analyzer=self.analyzer,
							    vocabulary=self.vocab,
							    ngram_range=ngram_range)

		doc_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(doc_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			doc_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			doc_features = model.get_feature_names()
			doc_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			doc_features = doc_features[:self.n_feats]

		new_X = []
		for feat in doc_features:
			for ft, vec in zip(model.get_feature_names(), doc_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		doc_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			doc_vectors = scaling_model.fit_transform(doc_vectors)

		return doc_vectors, doc_features, scaling_model

	# Term-Frequency Inverse Document Frequency Vectorization

	def tfidf(self, smoothing):

		# Define fed-in analyzer
		stop_words = self.stop_words
		ngram_range = ((self.ngram,self.ngram))

		model = TfidfVectorizer(stop_words=self.stop_words, 
							    max_features=self.n_feats,
							    analyzer=self.analyzer,
							    vocabulary=self.vocab,
							    ngram_range=ngram_range)

		tfidf_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(tfidf_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			tfidf_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			tfidf_features = model.get_feature_names()
			tfidf_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			tfidf_features = tfidf_features[:self.n_feats]

		new_X = []
		for feat in tfidf_features:
			for ft, vec in zip(model.get_feature_names(), tfidf_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		tfidf_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			tfidf_vectors = scaling_model.fit_transform(tfidf_vectors)
			
		return tfidf_vectors, tfidf_features, scaling_model

