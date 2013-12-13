# -*- coding: utf-8 -*-
from __future__ import division
import time, re, ast, numpy as np, pandas as p, cPickle as pickle
from scipy import sparse
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from scipy.cluster.vq import whiten
from nltk import clean_html, SnowballStemmer, PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

print 'Starting at '+time.strftime("%H:%M:%S", time.localtime())+'.'

### Create preprocessing functions 
# Word stemmer
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]
	if type is False or type not in supported_stemmers:
		return words_l
	else:
		l = []
		if type == "PorterStemmer":
			stemmer = PorterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "SnowballStemmer":
			stemmer = SnowballStemmer(lang)
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "LancasterStemmer":
			stemmer = LancasterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "WordNetLemmatizer":
			wnl = WordNetLemmatizer()
			for word in words_l:
				l.append(wnl.lemmatize(word).encode(encoding))
		return l

# String and tokenize
def preprocess_pipeline(str, stemmer_type="WordNetLemmatizer", lang="english", return_as_str=True, 
						do_remove_stopwords=True):
	l = []
	words = []
	
	# Tokenize
	sentences=[word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
			for t in sent_tokenize(str.replace("'", ""))]
	
	for sentence in sentences:
		# Remove stopwords
		if do_remove_stopwords:
			words = [w for w in sentence if w.lower() not in stopwords.words('english')]
		else:
			words = sentence
		# Stem words
		words = stemming(words, stemmer_type)
		
		# Convert to string
		if return_as_str:
			l.append(" ".join(words))
		else:
			l.append(words)
	if return_as_str:
		return " ".join(l)
	else:
		return l

# URL cleaner
def url_cleaner(url, stemmer_type="WordNetLemmatizer"):
        strip_list=['http', 'https', 'www', 'com', 'net', 'org', 'm', 'html', 'htm']
        url_list=[x for x in word_tokenize(" ".join(re.findall(r'\w+', url,
                flags = re.UNICODE | re.LOCALE)).lower()) if x not in strip_list
                and not x.isdigit() and x not in stopwords.words('english')]
        return " ".join(stemming(url_list, stemmer_type))

# Boilerplate extractor
def extract_boilerplate(str, stemmer_type="WordNetLemmatizer"):
        # Adjust 'null' and extract json
        try:
                json=ast.literal_eval(str)
        except ValueError:
                json=ast.literal_eval(str.replace('null', '"null"'))

        if 'body' in json and 'title' in json:
                return (preprocess_pipeline(json['title'], stemmer_type), preprocess_pipeline(json['body'], stemmer_type))
        elif 'body' in json:
                return ("", preprocess_pipeline(json['body'], stemmer_type))
        elif 'title' in json:
                return (preprocess_pipeline(json['title'], stemmer_type), "")
        else:
                return ("", "")

# TFIDF function
def process_tfidf(train, test, model):
        # Combine URL for TF-IDF
        grouped=train + test
        lentrain = len(train)

        # Fit the model according to training data
        print "Fitting TF-IDF pipeline"
        model.fit(grouped)

        # Transform matrix
        print "Transforming data"
        grouped = model.transform(grouped)
        tfidf_train = grouped[:lentrain]
        tfidf_test = grouped[lentrain:]

        return (tfidf_train, tfidf_test)
                
loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

# Def function to strip terms from 1 or 0 Y values
def IO_list(terms, yvals):
        term_dict_0={}
        term_dict_1={}
        tot_words_0=0
        tot_words_1=0
        all_words=[]
        # Add words to document, filtering on Y
        for page in range(len(yvals)):
                if yvals[page]==1:
                        for word in terms[page].split():
                                if word not in all_words:
                                        all_words.append(word)
                                if word in term_dict_1:
                                        term_dict_1[word]+=1
                                else:
                                        term_dict_1[word]=1
                                tot_words_1+=1
                elif yvals[page]==0:
                        for word in terms[page].split():
                                if word not in all_words:
                                        all_words.append(word)
                                if word in term_dict_0:
                                        term_dict_0[word]+=1
                                else:
                                        term_dict_0[word]=1
                                tot_words_0+=1                        
                else:
                        raise Exception("Y value is not 1 or 0")

        for t in term_dict_0:
                term_dict_0[t]/=tot_words_0
        for t in term_dict_1:
                term_dict_1[t]/=tot_words_1

        drop_list=[]
        # The 75% for each term dict is about .000005
        # The 95% for each term dict is about .00005 ??
        drop_val=.0003
        for word in all_words:
                if word in term_dict_1 and word in term_dict_0:
                        if term_dict_1[word]>drop_val and term_dict_0[word]>drop_val:
                                drop_list.append(word)
                
        return drop_list

# Load boilerplate text
print "Loading text"
traindata_raw = list(np.array(p.read_table('../data/train.tsv'))[:,2])
testdata_raw = list(np.array(p.read_table('../data/test.tsv'))[:,2])
y = np.array(p.read_table('../data/train.tsv'))[:,-1]
y=y.astype(int)

# Load URLs
train_url_raw = list(np.array(p.read_table('../data/train.tsv'))[:,0])
test_url_raw = list(np.array(p.read_table('../data/test.tsv'))[:,0])


# Preprocess URLs
print "Preprocessing URLs"

train_url = []
test_url = []
for observation in train_url_raw:
        train_url.append(url_cleaner(observation, "WordNetLemmatizer"))
for observation in test_url_raw:
        test_url.append(url_cleaner(observation, "WordNetLemmatizer"))

pickle.dump(train_url, open('Preprocessed train url.p', 'wb'))
pickle.dump(test_url, open('Preprocessed test url.p', 'wb'))

#train_url=pickle.load(open('Preprocessed train url.p', 'rb'))
#test_url=pickle.load(open('Preprocessed test url.p', 'rb'))

# Preprocess boilerplate
print "Preprocessing boilerplate"

train_body = []
test_body = []
train_title = []
test_title = []
for observation in traindata_raw:
        a, b=extract_boilerplate(observation, "WordNetLemmatizer")
        train_title.append(a)
        train_body.append(b)
for observation in testdata_raw:
        a, b=extract_boilerplate(observation, "WordNetLemmatizer")
        test_title.append(a)
        test_body.append(b)
		
pickle.dump(train_title, open('Preprocessed train title.p', 'wb'))
pickle.dump(test_title, open('Preprocessed test title.p', 'wb'))

pickle.dump(train_body, open('Preprocessed train body.p', 'wb'))
pickle.dump(test_body, open('Preprocessed test body.p', 'wb'))

#train_title=pickle.load(open('Preprocessed train title.p', 'rb'))
#test_title=pickle.load(open('Preprocessed test title.p', 'rb'))

#train_body=pickle.load(open('Preprocessed train body.p', 'rb'))
#test_body=pickle.load(open('Preprocessed test body.p', 'rb'))

# Create TF-IDF matrix parameters
tfv = TfidfVectorizer(min_df=15, max_features=None, strip_accents='unicode',  
		analyzer='word', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

# Create logit parameters
log = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
		C=1, fit_intercept=True, intercept_scaling=1.0, 
		class_weight=None, random_state=None)

# Group three data sets
comboX_train=[train_url[x] + ' ' + train_title[x] + ' ' + train_body[x] for x in range(len(train_body))]
comboX_test=[test_url[x] + ' ' + test_title[x] + ' ' + test_body[x] for x in range(len(test_body))]

# Gen stop word list for terms popular for both 1/0 Y values

new_stop_words=IO_list(comboX_train, y)
new_stop_words=set(new_stop_words)

# Remove common words from data sets
comboX_train=[' '.join(w for w in x.split() if w not in new_stop_words) for x in comboX_train]
comboX_test=[' '.join(w for w in x.split() if w not in new_stop_words) for x in comboX_test]

pickle.dump(new_stop_words, open('Common words.p', 'wb'))
pickle.dump(comboX_train, open('Train data with removed comm words.p', 'wb'))
pickle.dump(comboX_test, open('Test data with removed comm words.p', 'wb'))

#new_stop_words=pickle.load(open('Common words.p', 'rb'))
#comboX_train=pickle.load(open('Train data with removed comm words.p', 'rb'))
#comboX_test=pickle.load(open('Test data with removed comm words.p', 'rb'))

print "Number of common words: {}".format(len(new_stop_words))

# Process and transform TF-IDF for variable groups
tfidf_combo_train, tfidf_combo_test=process_tfidf(comboX_train, comboX_test, tfv)

# Test CV score
print "Pre-transform log 10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(
	log, tfidf_combo_train, y, cv=10, scoring='roc_auc'))

# Transform data
print "Transforming data"
log.fit(tfidf_combo_train,y)
tfidf_combo_train=log.transform(tfidf_combo_train)
tfidf_combo_test=log.transform(tfidf_combo_test)

print "Shape: ", tfidf_combo_train.shape

# Test CV score
print "Post-transform log 10 Fold CV Score: ", np.mean(cross_validation.cross_val_score(
	log, tfidf_combo_train, y, cv=10, scoring='roc_auc'))

# Run logit
print "training on full data"
log.fit(tfidf_combo_train,y)

pred_train = log.predict_proba(tfidf_combo_train)[:,1]
pred_test = log.predict_proba(tfidf_combo_test)[:,1]

# Write out files
train_lab = p.read_csv('../data/train.tsv', sep="\t", na_values=['?'], index_col=1)
pred_file = p.DataFrame(pred_train, index=train_lab.index, columns=['label'])
pred_file.to_csv('train sans common words.csv')

test_lab = p.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
test_file = p.DataFrame(pred_test, index=test_lab.index, columns=['label'])
test_file.to_csv('real sans common words.csv')

print 'Finished at '+time.strftime("%H:%M:%S", time.localtime())+'.'
