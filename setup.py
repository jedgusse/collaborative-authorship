try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import sys

if sys.version_info[0] == 2:
    ete_requirement = 'ete2'
elif sys.version_info[0] == 3:
    ete_requirement = 'ete3'

setup(
	name = 'go',
	version = '1.0',
	# Python modules
	author = 'Jeroen De Gussem',
	author_email = 'jedgusse.degussem@ugent.be',
	py_modules = [''],
	# Dependencies
	install_requires=[
		'Click',
		'numpy',
        'scikit-learn',
        'seaborn',
        'nltk',
        'matplotlib',
	],
	# Instruction for set-up tools
)
