# box-office
This project takes a Bayesian approach to predicting box office revenue for movies. We leveraged data from Rotten Tomatoes to form initial predictors in our Bayesian regression, but we also engineered additional features by extracting latent themes from pre-release movie reviews. This would test our hypothesis that certain types of movies, outside of general (well-codified) genres, would generally be bigger box office hits than others. We used Latent Dirichlet Allocation to uncover latent topics within the movie reviews. To include these in our regression, we added a feature for each latent topic representing the probability that a given movie belonged to that topic. Details of the entire analysis are explained thoroughly in the PDF in the repo.

The following packages are needed to run the ipynb file:

import numpy as np   
import pandas as pd  
from sklearn.decomposition import LatentDirichletAllocation  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
import nltk  
from nltk.corpus import stopwords  
import json  
from sklearn.impute import SimpleImputer  
import matplotlib.pyplot as plt  
from matplotlib import animation  
import pymc as pm  
import arviz as az  
import graphviz  
import seaborn as sns  

This notebook can be run in JupyterLab or Jupter Notebook.
