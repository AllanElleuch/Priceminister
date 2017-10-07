import pandas
import csv
import random
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

class toolKit:
    def __init__(self, dataframe=None):
        self.dataFrame=dataframe

        ## filename : string. import a csv as dataframe
    def addDataframe(self, filename):
        if(self.dataFrame is None):
            self.dataFrame=open_with_pandas_read_csv(filename)
        else:
            self.dataFrame=pandas.merge(self.dataFrame, open_with_pandas_read_csv(filename))

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
        # plt.plot(values0,counts0,'b',label='Commentaire inutiles')
        # plt.plot(values1,counts1,'r',label='Commentaire utiles')
        plt.plot(self.dataFrame.len,self.dataFrame.Target,'ro',label='Commentaire utiles')
        plt.ylabel('helpfulness')
        plt.xlabel('review length')
        plt.suptitle('product review length')

        plt.show()
    def entrainerModelRandomForest(self):
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        # b=np.array(tableauInputOutput['len']).reshape((-1,1))
        # c=np.array(tableauInputOutput['Target']).reshape((-1,1))
        # regr.fit(, self.dataFrame['Target'])
        print(self.dataFrame)

csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def open_with_pandas_read_csv(filename):
    return pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)


def open_with_pandas_read_csv2(filename, delimiter):
    return   pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)
