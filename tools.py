import pandas
import csv
import random
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from scipy import sparse
from sklearn import linear_model, pipeline, feature_extraction
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from features import FeaturesCalculator as fcalculator

class toolKit:
    def __init__(self, dataframe=None,randomForest=True):
        self.dataFrame=dataframe
        if(randomForest):
            self.regr=RandomForestRegressor(n_jobs = -1,random_state =None,n_estimators=50,max_depth= None)
            # self.regr=RandomForestRegressor(n_jobs = -1,random_state =None, max_depth= 1,n_estimators=15)
        else:
            self.regr=svm.SVC()
        self.X_train_tfidf=None
        self.tfidf_matrix=None
        self.count_vect = CountVectorizer(analyzer='word',stop_words=frozenset(stopwords.words('french')) ,max_df=0.8,n_features=3000,ngram_range=(1,2) ) #regarde mot qui existe
        self.tfidf_transformer = TfidfTransformer() # use_idf=True, smooth_idf=False, sublinear_tf=False



# voir Normalized review text length
# et https://en.wikipedia.org/wiki/Normalization_(statistics)
# Readability (ARI) 0.21

# For length we used 4 features. We used the text length in
# characters which we normalized by dividing it by 1000
# and three binary variable indicating whether a review
# was short, medium or long. We hoped that the binary
# variables would capture the thresholds we saw in the
# exploratory analysis (Figures 2, 3).
# • binary: Short review? ( < 100 characters)
# • binary: Medium review? (100 to 1000 characters)
# • binary: Long review? ( > 1000 characters)

# *Number of title words 0.2535
# *Number of total punctuation 0.2327
#
# Deviation from mean review score 0.4688 (reviewer rating - meanrating)
# Review score 0.5431
#DO Absolute deviation correlation

    def addColLen(self):
        listLen= self.dataFrame['review_content'].values.tolist()
        tabcat=[]


        for review in listLen:
            reviewSize=len(review)
            categorie=0
            if (reviewSize<250):
                categorie=1
            if (reviewSize < 1600):
                categorie=2
            tabcat.append(categorie)

        self.dataFrame['cat']= np.array(tabcat)
        self.dataFrame['len']= self.dataFrame['review_content'].str.len()
        # self.dataFrame['len']= self.dataFrame['review_title'].str.len()
        self.dataFrame['lenTitle']= self.dataFrame['review_title'].str.split().str.len()

    def addColLenPonctuation(self):
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        self.dataFrame['lenPonctuation']= [count(a, punctuation) for a in  self.dataFrame['review_content'].values.tolist() ]

    def addTFIDF(self):
        self.dataFrame['len']= self.dataFrame['review_content'].str.len()
        vectorizer = CountVectorizer( analyzer=SpecialAnalyzer()) # SpecialAnalyzer is my class extracting features from text
        vectorizer = CountVectorizer( analyzer=SpecialAnalyzer()) # SpecialAnalyzer is my class extracting features from text




    def add_features_dataframe(self,lambdaFunction,columnLabel):
        data = self.dataFrame.apply (lambda row: lambdaFunction(row),axis=1)
        self.dataFrame[columnLabel] = data
        return

    def TFIDF(self,transformOnly=False):

        # data = self.dataFrame['review_title']
        # print(self.dataFrame)
        data = self.dataFrame['review_content']
        # data = np.concatenate( data, axis=0 )
        dataFiltered = []
        # stopWords = list(stopwords.words('french'))
        # for line in data:
        #     for w in data:
        #         if w not in stopWords:
        #             dataFiltered.append(w)

        # tab = []
        # for line in data:
        #     tab.append(train_tfidf(line))
        # self.dataFrame['tfidf']=np.array(tab)


        # count_vect,tfidf_transformer, X_train_tfidf = self.train_tfidf(data)
        count_vect,tfidf_transformer, X_train_tfidf = self.train_tfidf(data,transformOnly)
        print(X_train_tfidf)
        self.tfidf_matrix=X_train_tfidf
        self.dataFrame['tfidf']=X_train_tfidf

        self.X_train_tfidf=X_train_tfidf

    def train_tfidf(self,doc,transformOnly=False):

        # count_vect = CountVectorizer(analyzer='word',stop_words=frozenset(stopwords.words('french')) ,max_features=1000,min_df=1 ) #regarde mot qui existe
        # count_vect = CountVectorizer(analyzer='word',stop_words=None ,max_features=1000,min_df=1 ) #regarde mot qui existe
        # tfidf_transformer = TfidfTransformer()

        # print(X_train_counts.shape)
        if transformOnly:
            X_train_counts = self.count_vect.transform(doc)

            X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        else:
            X_train_counts = self.count_vect.fit_transform(doc)

            X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        # print(X_train_tfidf.shape)
        return [self.count_vect,self.tfidf_transformer, X_train_tfidf]

    def train_tfidf_hash(self,doc):
        def token_freqs(doc):
            """Extract a dict mapping tokens from doc to their frequencies."""
            freq = defaultdict(int)
            for tok in tokens(doc):
                freq[tok] += 1
            return freq
        count_vect = FeatureHasher(n_features=1000) #regarde mot qui existe
        X_train_counts = count_vect.fit_transform(doc)
        print(X_train_counts.shape)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # print(X_train_tfidf.shape)
        return [count_vect,tfidf_transformer, X_train_tfidf]

        # nltk.download('stopwords')
        # print(stopWords)
        # print("counts")
        # print(X_train_counts)
        #
        # print("features : ")
        # print(count_vect.get_feature_names()[::-1])
        # tab=[]
        # for i in data.values.tolist():
        #     tab+=i

        # print(tab)
        # print(data)

        ## filename : string. import a csv as dataframe
    def addDataframe(self, filename, newFrame=False):
        if(self.dataFrame is None or newFrame == True):
            self.dataFrame=open_with_pandas_read_csv(filename)
        else:
            self.dataFrame=pandas.merge(self.dataFrame, open_with_pandas_read_csv(filename))
            # self.dataFrame=pandas.concat([self.dataFrame, open_with_pandas_read_csv(filename)], axis=1)


    def drawGraphReviewstars(self):
        tableauRepartitionReviewStars=[0,0,0,0,0]
        tableauReviewStars = self.dataFrame['review_stars'].values.tolist()
        for i in range(len(tableauReviewStars)):
            index = tableauReviewStars[i]
            tableauRepartitionReviewStars[index-1] +=1
        names = ['1 stars', '2 stars', '3 stars', '4 stars', '5 stars']
        fig, ax = plt.subplots()
        plt.bar([1,2,3,4,5], tableauRepartitionReviewStars)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(names)
        plt.suptitle('product review stars')
        plt.show()

# draw a graph with review length as x and helpfulness as y
    def drawGraphReviewlength(self):
        # test = np.where(df['Target']==1,df['len'],None)
        # print(df.loc[df['Target'] == 1]['Target'])
        df=self.dataFrame
        values0, counts0 = np.unique(df.loc[df['Target'] == 0].len.tolist(), return_counts=True)
        values1, counts1 = np.unique(df.loc[df['Target'] == 1].len.tolist(), return_counts=True)
        print(self.dataFrame)
        plt.plot(values0,counts0,'b',label='Commentaire inutiles')
        plt.plot(values1,counts1,'r',label='Commentaire utiles')
        # plt.plot(self.dataFrame.len,self.dataFrame.Target,'ro',label='Commentaire utiles')
        plt.ylabel('helpfulness')
        plt.xlabel('review length')
        plt.suptitle('product review length')

        plt.show()
    def entrainerModelRandomForest(self,parameter=['len']): #TODO : Faire cross validation
        # regr = RandomForestRegressor(n_jobs = -1,random_state =None, max_depth= 70,n_estimators=10)

        if( False ):
            print("no tfidf")
            regr.fit(self.dataFrame[parameter].values, self.dataFrame['Target'].values)
            self.regr=regr
        else:

            tfidf_matrix = self.tfidf_matrix
            new_column = self.dataFrame[parameter]
            final = scipy.sparse.hstack((tfidf_matrix, new_column))
            self.regr.fit(final, self.dataFrame['Target'].values )


    def setFeatures(self):
        self.add_features_dataframe(fcalculator.features_tokenized_content,'tokenized_content')
        self.add_features_dataframe(fcalculator.features_tokenized_title,'tokenized_title')
        self.add_features_dataframe(fcalculator.features_difficultword_content,'difficultword_content')
        self.add_features_dataframe(fcalculator.features_linsear_title,'linsear_title')
        return

    def crossvalidation(self):


        from sklearn.metrics import roc_auc_score
        from sklearn import metrics
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score),'Recall': make_scorer(recall_score),'f1': make_scorer(f1_score)}

        scores2 = cross_validate(self.regr ,X= self.dataFrame,y=self.dataFrame['Target'],cv=5,scoring=scoring) # return_train_score=False,
        print(scores2)
        Report = ""

        report+="Cross validate in 5 k mean \n"
        report+="mean fit time " + str( mean( scores2['fit_time']))   ) +"\n"
        report+="mean test accuracy  " + str( mean( scores2['test_Accuracy']))   )+"\n"
        report+="mean test_Recall  " + str( mean( scores2['test_Recall']))   )+"\n"
        report+="mean test_f1  " + str( mean( scores2['test_f1']))   )+"\n"
        report+="mean test aur_roc  " + str( mean( scores2['test_AUC']))   )+"\n"
        print(report)
        save =   open(os.path.join(os.path.dirname(__file__), "gridsearch.txt"), 'w')
        save.write(report)
        # save.write(bestResults+"\n")
        return

    def gridsearch(self):
        param_grid = {
            'n_estimators': [200,300,400],
            "min_samples_split" : [2,4], #def 2
            "bootstrap": [True, False], #true
              "min_samples_split": [2, 5,10], # 2
               "max_depth": [75, 125,None],
            "max_features": ['auto', 'sqrt', 'log2']
        }
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}


        CV_rfc = GridSearchCV(estimator=clf,
                          param_grid=param_grid,
                          scoring=scoring, cv=5, refit='AUC') #, refit='AUC'
        CV_rfc.fit(featuresTest, y_true_merge)
        results = str(CV_rfc.cv_results_)
        results = "cv_results_ : " + results +"\n"
        bestResults = str(CV_rfc.best_score_ )
        bestResults = "best_score_ : " + bestResults
        bestParam = str(CV_rfc.best_params_)+"\n"
        bestParam = "best_params_ : " + bestParam +"\n"
        print(bestResults)
        print("Best parameters : ")
        print(bestParam)


        save =   open(os.path.join(os.path.dirname(__file__), "crossValidation.txt"), 'w')
        save.write(bestParam+"\n")
        save.write(bestResults+"\n")

    def print_corrcoef(self,colTarget=None):
        columns = self.dataFrame.columns.values
        forbiddenColumn = ['Target','review_content','review_title','product']
        print(columns)
        if not colTarget:
            for col in columns:
                if(col not in forbiddenColumn ):
                        print("Correlation of "+col +" "+ str(np.corrcoef(self.dataFrame['Target'],self.dataFrame[col])[0][1])   )
        else:
            print("Correlation of "+colTarget +" "+ str(np.corrcoef(self.dataFrame['Target'],self.dataFrame[colTarget])[0][1])   )





    def testRegrEfficiency(self,parameter):
        df = self.dataFrame
        if(self.regr != None):
            # tabInput=self.dataFrame[parameter].as_matrix()
            # tfidf=list(self.X_train_tfidf.toarray())
            # tab = np.concatenate(tabInput,tfidf)
            # tabGuess=(self.regr.predict(tab))
            # tabGuess=(self.regr.predict(df[parameter].values))
            tabInput=self.dataFrame[parameter]
            tfidf=self.X_train_tfidf

            tfidf_matrix = self.tfidf_matrix
            new_column = self.dataFrame[parameter]
            # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            # new_column = sel.fit_transform(new_column)
            final = scipy.sparse.hstack((tfidf_matrix, new_column))

            # tabInput= np.concatenate(tabInput,tfidf)
            # tabGuess=(self.regr.predict(tabInput))
            # tabGuess=(self.regr.predict(df[parameter].values))
            tabGuess=self.regr.predict(final)
            # tabGuess=self.regr.predict(tfidf_matrix)

            output = df['Target'].values.tolist()
            # print("tabGuess")

            # print(tabGuess)
            score = 0
            # print(tabGuess)
            # print(output)
            for i in range(len(tabGuess)):
                accuracy = 0.50
                if (output[i]==1 and tabGuess[i] >accuracy) or (output[i]==0 and tabGuess[i] <=accuracy) :
                    score+=1
            # print(score)
            score = score/len(tabGuess)
            print( "auto score of current model : "+ str(score))
        else :
            print("regr is not defined")

    def clear(self):
        self.dataFrame=None
    def getFeatures():
        self.addColLen()
        self.addColLenPonctuation()
        parameter=['len','cat','review_stars','lenTitle', 'lenPonctuation']
        self.TFIDF()
    # Do the csv to upload on the challenge website
    def testForChallenge(self,parameter):
        self.addDataframe('input_test.csv',newFrame=True)
        self.setFeatures()
                # self.addColLen()
                # self.addColLenPonctuation()
        path = os.path.join(__location__, './output/output_for_challenge.csv')
        if(self.tfidf_matrix):
            self.TFIDF(transformOnly=True)

        tfidf_matrix = self.tfidf_matrix
        # print(self.dataFrame)
        new_column = self.dataFrame[parameter]
        final = scipy.sparse.hstack((tfidf_matrix, new_column.values))
        # regr.fit(final, self.dataFrame['Target'].values )
        tabGuess=(self.regr.predict(final))


        # self.clear()
        # self.addDataframe('input_test.csv')
        # self.addColLen()
        # parameter=['len','review_stars']
        # # self.entrainerModelRandomForest(parameter)
        # # tfidf=list(self.X_train_tfidf.toarray())
        # # tab = np.concatenate(tabInput,tfidf)
        # # tabGuess=(self.regr.predict(tab))
        # tabGuess=(self.dataFrame['tfidf'].tolist(),self.dataFrame[parameter].values)
        tabID= self.dataFrame['ID'].values.tolist()
        if(len(tabID)==len(tabGuess)):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID;Target"])

                for i in range(len(tabGuess)):
                    if tabGuess[i]>0.5:
                        writer.writerow([(str(tabID[i])+';1' )])
                    else:
                        writer.writerow([(str(tabID[i])+';0' )])

                    # print((str(x)+";"+str(random.randint(0, 1))));
                print("done")



csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def open_with_pandas_read_csv(filename):
    pd = pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)
    return pd


def open_with_pandas_read_csv2(filename, delimiter):
    return   pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)
