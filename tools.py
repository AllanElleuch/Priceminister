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

    def drawGraphReviewlength(self):
        listInputReviewContent = self.dataFrame[['review_content','Target']]
        TabReviewLength =[]

        listInputReviewContent.groupby('Target').plot.hist()

        for review in listInputReviewContent['review_content'].values.tolist():
            TabReviewLength.append(len(review))
        # print(TabReviewLength)

        dicoReviewLength={}
        for i in TabReviewLength:
            if not i in dicoReviewLength:
                dicoReviewLength[i]=1
            else:
                dicoReviewLength[i]+=1
        x,y=[],[]

        for key,value in dicoReviewLength.items():
            x.append(key)
            y.append(value)

        print(dicoReviewLength)


        valx=x
        values = y
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
        print(r_value)
        fig, ax = plt.subplots()
        plt.bar(valx, values)
        plt.suptitle('product review length')
        plt.ylabel('Number of reviews')
        plt.xlabel('review length')

        plt.show()

csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def open_with_pandas_read_csv(filename):
    return pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)


def open_with_pandas_read_csv2(filename, delimiter):
    return   pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)
