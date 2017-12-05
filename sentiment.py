from nltk.corpus import sentiwordnet as swn
import nltk
nltk.download('wordnet')
# python -m nltk.downloader sentiwordnet nltk.download('punkt')
# python -m nltk.downloader  nltk.download('punkt')
# python -m nltk.downloader  nltk.download('punkt')

# python -m nltk.downloader  punkt
# python -m nltk.downloader  averaged_perceptron_tagger
# python -m nltk.downloader  sentiwordnet

# breakdown = swn.senti_synset('genial.a.01')
# print(breakdown)
# breakdown.pos_score()
# breakdown.neg_score()
# breakdown.obj_score()


import numpy as np
import pandas
import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def path(filename):
    return  os.path.join(os.path.dirname(__file__), filename)

pd = pandas.read_csv(path('input_test.csv'),sep=';')
# pd = pandas.read_csv(path('trained_data.csv'))
# pd = pandas.read_csv(path('trained_data.csv'))[0:1]

def review_Pruning(data):
    # data = row['review_content']
    if(data == None or len(data)<=1):
        return ""
    else:
        return data

# data = pd.apply (lambda row: lambdaFunction(row),axis=1)
print('Document prunning process')

# pd['review_content'] = pd['review_content'].map(review_Pruning)
# print(pd['review_content'])

sentences = pd['review_content']

# for ligne in pd['review_content']:
#     if len(ligne) <= 1 or ligne == None:
#         print("ligne : " + ligne)
tagMap = {
'ADJ' : 's', # adjective - ADJECTIVE
'NN':'n',
'VB':'v',
'JJ':'a',
# :'s',
'ADV':'r'
 }
# s - ADJECTIVE SATELLITE
# r - ADVERB
# n - NOUN
# v - VERB
# ADJ	adjective	new, good, high, special, big, local
# ADP	adposition	on, of, at, with, by, into, under
# ADV	adverb	really, already, still, early, now
# CONJ	conjunction	and, or, but, if, while, although
# DET	determiner, article	the, a, some, most, every, no, which
# NOUN	noun	year, home, costs, time, Africa
# NUM	numeral	twenty-four, fourth, 1991, 14:24
# PRT	particle	at, on, out, over per, that, up, with
# PRON	pronoun	he, their, her, its, my, I, us
# VERB	verb	is, say, told, given, playing, would
# .	punctuation marks	. , ; !
# X
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNPS']
    # return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN',  'VBZ','VBP']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    else:
         # return ''
         return None

from googletrans import Translator
translator = Translator(service_urls=[
      'translate.google.ca','translate.google.com'

    ])

def translateOneText(texte):
    translations = translator.translate(texte, dest='en')
    return translations.text


# translations =translator.translate(doc, dest='en')

translated=[]
doc = list(sentences.values)

def translateDataframeToFile(doc):
    i=0
    for texte in doc:
        i+=1
        print("Texte : " + str(i))
        try:
            traduit = translateOneText(texte)
            translated.append(traduit)
        except Exception as e:
            translated.append("")

    english = pandas.Series(translated)
    pd['english'] =english.values
    pd.to_csv('./eng_data_testing.csv', encoding='utf-8',index=False)
    # pd.to_csv('./eng_data_training.csv', encoding='utf-8',index=False)

translateDataframeToFile(doc)

raise

print('Ask google to translate')
print('Translation finnished')

# translations = ['It would rather be a movie very well if they did not engage Richard Dreyfus. The young and plausible Liv Taylor and John Lithgow are superb but the pompous Richard Dreyfus ruins a lot. Pity']

# print(doc)
# print(type(doc))

# lemmatized HERE


# translations = translator.translate(doc, dest='ko')

# sentences_translated =
# print(translations)

from nltk.corpus import wordnet as wn
class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return None
            # return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        # pos_tokens = []



        res = []
        for pos in pos_tokens:
            for (word,pos_tag) in pos:
                tag = self.get_wordnet_pos(pos_tag)
                # if (True ):
                # print(tag)
                if(tag is not None):
                    a =(lemmatizer.lemmatize(word,tag), [pos_tag])
                    # print(a)
                    res.append(a)
        # pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]

        return res
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
print("Start positive negative process")
tab = []
tabPos = []
tabNeg=[]
tabEnglish=[]
i=0
for item in translations:
    print("NUMERO : " + str(i))
    i+=1
    sentence = item.text
    tabEnglish.append(sentence)
    print(sentence)
    # lemmatizer = WordNetLemmat izer()
    # splitter = Splitter()
    # lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()
    #
    # #step 1 split document into sentence followed by tokenization
    # tokens = splitter.split(sentence)
    #
    # print(tokens)
    # # #step 2 lemmatization using pos tagger
    # lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)
    # print(lemma_pos_token)
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem import WordNetLemmatizer
    porter_stemmer = PorterStemmer()

    wordnet_lemmatizer = WordNetLemmatizer()
    stop = stopwords.words('english') + list(string.punctuation)
    token = [i for i in word_tokenize(sentence) if i.lower() not in stop ]
    token = [wordnet_lemmatizer.lemmatize(i) for i in word_tokenize(sentence) if i.lower() not in stop ]
    # token = [ porter_stemmer.stem(i) for i in sentence]

    # token = [i for i in word_tokenize(sentence.lower()) if i not in stop ]

    # print("stem : ")
    # print(token)
    tagged = nltk.pos_tag(token)
    # print(tagged)
    tagged = [(i,penn_to_wn(i[1])) for i in tagged if penn_to_wn(i[1]) is not None  ]

    # print(token)
    # print(tagged)
    # print(tagged)
    breakdown = swn.senti_synset('superb.a.01')
    # print(breakdown)
    pos = 0
    neg = 0
    for (word,tag),wordnetTag in tagged:
        try:
            breakdown = swn.senti_synset(word+'.'+wordnetTag+'.01')
            # print(breakdown)
            pos +=breakdown.pos_score()
            neg += breakdown.neg_score()
        except Exception as e:
            print("ERROR :" + word)
    # tab.append((pos,neg))
    tabPos.append(pos)
    tabNeg.append(neg)




listpos = pandas.Series(tabPos)
listneg = pandas.Series(tabNeg)
english = pandas.Series(tabEnglish)
pd['english'] =english.values
pd['listpos'] =listpos.values
pd['listneg'] =listneg.values
print(pd)
pd.to_csv('./eng_data_training.csv', encoding='utf-8',index=False)
print()

                # breakdown.pos_score()
                # breakdown.neg_score()

        # wordnet_lemmatizer.lemmatize(‘dogs’)
    # lemma = nltk.stem.WordNetLemmatizer()
    # listLemma=[]
    # for word,tag in tagged:
    #     print(tag)
    #     print(word)
    #     cat = penn_to_wn(tag)
    #     print("cat : "+cat)
    #     l = lemma.lemmatize(word,cat )
    #     listLemma.append(l)
    # print(listLemma)
# for item in translations:
#     sentence = item.text
#     token = nltk.word_tokenize(sentence)
#     tagged = nltk.pos_tag(token)

    # print(tagged)
    # lemma = nltk.stem.WordNetLemmatizer()
    # listLemma=[]
    # for word,tag in tagged:
    #     print(tag)
    #     cat = penn_to_wn(tag)
    #     print("cat : "+cat)
    #     l = lemma.lemmatize('loving',cat )
    #     listLemma.append(l)
    # print(listLemma)


    # wordOfInterest = []
    # for word,cat in tagged:
    #     # print(word +" " + cat)
    #
    #     d = wn.synsets('word')
    #     print(d)
    #     print(cat)
    #
    #
    #     if(cat in tagMap.keys()):
    #         # print("add " + word)
    #         wordOfInterest.append((word,tagMap[cat]))
        # else:
            # print("CAT NON GERER : " + cat +" " + word)
#     for word,cat in wordOfInterest:
#         print(word +" -----> " + cat)
# #         # print("hello")
#         breakdown = swn.senti_synset(word+'.'+cat+'.01')
#         print(breakdown)



        # print(breakdown)
        # breakdown.pos_score()
        # breakdown.neg_score()
        # breakdown.obj_score()
# list(swn.senti_synsets('slow'))
# happy = swn.senti_synsets('happy', 'a')
# print( swn.all_senti_synsets())
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# sentence = "Iphone6 camera is awesome for low light "
# token = nltk.word_tokenize(sentence)
# print(token)
# tagged = nltk.pos_tag(token)
# print(tagged)

# breakdown = swn.senti_synset('breakdown.n.03')
