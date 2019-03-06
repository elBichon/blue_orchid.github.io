import argparse
import wikipedia
import bs4 as bs 
import urllib.request
from bs4 import BeautifulSoup
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

####################
## this function takes as input the language and the subject you want to create a classifier for
## it will then get the ad hoc url in wkipedia
## then it will scrape each span to get its title as a category
####################
def get_category(language, subject):
    wikipedia.set_lang(language)
    source = urllib.request.urlopen(wikipedia.page(subject).url).read()
    soup = bs.BeautifulSoup(source,'lxml')
    soup_txt = str(soup.body)
    category = []
    for each_span in soup.find_all('span', {'class':'mw-headline'}):
        soup = BeautifulSoup(str(each_span).replace(' ','_'), "html.parser").getText()
        category.append(soup)
    return category

####################
## this function takes as input the language and the subject 
## it gets the text for each span and cleans it 
## then it will output a list containing each clean sentence
####################
def get_data(language, subject):
    wikipedia.set_lang(language)
    source = urllib.request.urlopen(wikipedia.page(subject).url).read()
    soup = bs.BeautifulSoup(source,'lxml')
    soup_txt = str(soup.body)
    div = []
    for each_span in soup.find_all('span', {'class':'mw-headline'}):
        str(each_span).replace(' ','_')
        div.append(str(each_span))
    filter_tag = []
    i = 0
    while i < len(div)-1:
        start = div[i]
        end = div[i+1]
        text = soup_txt[soup_txt.find(start)+len(start):soup_txt.rfind(end)]
        soup = str(BeautifulSoup(text, "html.parser"))
        soup = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", soup)
        soup = BeautifulSoup(soup, "html.parser")
        soup = re.compile(r'<img.*?/>').sub('', str(soup.find_all('p')))
        soup = BeautifulSoup(soup, "html.parser")
        soup = (re.sub("[^a-zA-Z,.;:!0-9]"," ",soup.getText()).replace('[','').replace(']','').lstrip().rstrip().lower())    
        clean_text = re.sub(' +', ' ',soup).replace(',',' ')
        filter_tag.append(clean_text)        
        i += 1
    return filter_tag

####################
## this function takes as input the list of filter tags
## it will output a list containing the number of sentences in each category
####################
def get_len_list(filter_tag):
    filtered_text = []
    len_list = []
    i = 0 
    while i < len(filter_tag):
        doc = nlp(filter_tag[i])
        text = [sent.string.strip() for sent in doc.sents]
        filtered_text.append(text)
        len_list.append(len(filtered_text[i]))
        i += 1
    return filtered_text,len_list

####################
##tegory this function takes as input the len_list and the list of category
## it will output a dataframe containing the label (category) for each sentence
####################
def generate_dataset(len_list, category):
    i = 0
    label_list = []
    while i < len(len_list):
        j = 0
        if len_list[i] != 0:
            while j != len_list[i]:
                label_list.append(category[i].lower())
                j += 1
        i += 1
    flat_list = [item for sublist in  get_len_list(get_data(language, subject))[0] for item in sublist]
    data = {'text': flat_list,'label': label_list}
    df = pd.DataFrame.from_dict(data)
    print(df.head())
    print('Repartition of labels:', df['label'].iloc[0])
    print('Data Shape:', df.shape)
    return df

####################
## this function takes as input the language 
## it will output the list of stopwords for this language as a list 
####################
def get_stop_words(language):
    if language == 'en':
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    if language == 'fr':
        spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
    if language == 'de':
        spacy_stopwords = spacy.lang.de.stop_words.STOP_WORDS
    if language == 'es':
        spacy_stopwords = spacy.lang.es.stop_words.STOP_WORDS
    if language == 'pt':
        spacy_stopwords = spacy.lang.pt.stop_words.STOP_WORDS
    if language == 'it':
        spacy_stopwords = spacy.lang.it.stop_words.STOP_WORDS
    if language == 'nl':
        spacy_stopwords = spacy.lang.nl.stop_words.STOP_WORDS
    return spacy_stopwords

####################
## this function takes as input the language on the generated dataset
## it performs some last cleaning and data visualsation on the dataset
####################
def clean_dataset(language, df):
    srce_labels = df.label.values.tolist()
    srce_text = df.text.values.tolist()
    spacy_stopwords = get_stop_words(language)
    clean_text = []
    text = []
    i = 0
    while i < len(srce_text):
        extract = []
        doc = nlp(srce_text[i])
        for token in doc:
            extract.append(token.lemma_)
        clean_text.append(",".join(extract).replace(","," ").replace("   "," "))
        i += 1
    print('Number of stop words: %d' % len(spacy_stopwords))
    i = 0
    while i < len(clean_text):
        doc = nlp(clean_text[i])
        tokens = [token.text for token in doc if not token.is_stop]
        text.append(",".join(tokens).replace(","," ").replace("  "," ").replace("    "," ").replace("-PRON-"," ").rstrip().lstrip())
        i += 1
    data = {'text': text,'label': srce_labels}
    df = pd.DataFrame.from_dict(data)
    df = df.dropna()
    return df

####################
##this function takes as input the clean dataset
## using bag of words model it will create a random forest model for classification 
####################
def create_model(df):
    print("Creating the bag of words...\n")
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000) 
    train_data_features = vectorizer.fit_transform(df.text.values.tolist())
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)
    print("Training the random forest...")
    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit(train_data_features, df["label"])
    filename = language+"_"+subject.replace(" ","_")+'.sav'
    pickle.dump(forest, open(filename, 'wb'))
    print('saving model as: '+filename)

####################
## this function takes as input the dataset, model and text you want to classify
## it will vectorise the new given sentence and classify it usig the model you chose
####################
def call_model(csv_file, model_file, text):
    extract = []
    clean_text = []
    clean_test_reviews = []
    spacy_stopwords = get_stop_words(language)
    print("Creating the bag of words...\n")
    df = pd.read_csv(csv_file) 
    df = df.dropna()
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000) 
    train_data_features = vectorizer.fit_transform(df.text.values.tolist())
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)
    doc = nlp(text)
    for token in doc:
        extract.append(token.lemma_)
        clean_text.append(",".join(extract).replace(","," ").replace("   "," "))
    print('Number of stop words: %d' % len(spacy_stopwords))
    i = 0
    while i < len(clean_text):
        doc = nlp(clean_text[i])
        tokens = [token.text for token in doc if not token.is_stop]
        i += 1
    clean_test_reviews.append(",".join(tokens).replace(","," ").replace("  "," ").replace("    "," ").replace("-PRON-"," ").rstrip().lstrip())
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    loaded_model = pickle.load(open(model_file, 'rb'))
    result = loaded_model.predict(test_data_features)
    return result



parser = argparse.ArgumentParser(description='unin running')
parser.add_argument('request', type=str, help='choosing between predicting or training')
parser.add_argument('subject', type=str, help='subject of the classifier/name of the model')
parser.add_argument('language', type=str, help='language of the classifier/dataset to use')
args = parser.parse_args()
request  = args.request
subject = args.subject
language = args.language

nlp = spacy.load(language)

if request == 'predict':
	file_name = language+"_"+subject.replace(" ","_")+'.csv'
	model = language+"_"+subject.replace(" ","_")+'.sav'
	#to_predict = raw_input('sentence to classify')
	to_predict = input('sentence to classify')
	result = call_model(file_name, model, to_predict)
	print(result)

elif request == 'train':
	df = generate_dataset(get_len_list(get_data(language, subject))[1], get_category(language, subject))
	df = clean_dataset(language, df)
	file_name = language+"_"+subject.replace(" ","_")+'.csv'
	print('saving dataframe as: ',file_name)
	df.to_csv(file_name, sep=',', encoding='utf-8')
	fig = plt.figure(figsize=(8,4))
	sns.barplot(x = df['label'].unique(), y=df['label'].value_counts())
	plt.show()
	forest = create_model(df)

