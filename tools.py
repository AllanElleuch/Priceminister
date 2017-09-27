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

        self.dataFrame['len'] = [len(x) for x in self.dataFrame['review_content']]

        # print(self.dataFrame)
        TabReviewLength =self.dataFrame['len'].values.tolist()

        df=self.dataFrame
        test = np.where(df['Target']==1,df['len'],None)
        # print(df.loc[df['Target'] == 1]['Target'])

        print(self.dataFrame.groupby(['Target']).len.mean() )

        # colors = {'D':'red', 'E':'blue', 'F':'green', 'G':'black'}

        # drinks.groupby('continent').mean().plot(kind='bar')
        # grouped = df.groupby('color')
        # for key, group in grouped:
        #     group.plot(ax=ax, kind='scatter', x='carat', y='price', label=key, color=colors[key])
        #
        # plt.show()

        # dicoReviewLength = {i : 1 if 1==1 else i:1  for i in TabReviewLength  }

        # transforme un tableau en un tuple valeur : nombre d'occurence
        dicoReviewLength={}
        for i in TabReviewLength:
            dicoReviewLength[i] =   dicoReviewLength[i]+1 if i in dicoReviewLength else 1
        x,y = list(dicoReviewLength.keys()),list(dicoReviewLength.values())

        # x,y=[],[]
        # print(TabReviewLength)
        # for key,value in dicoReviewLength.items():
        #     x.append(key)
        #     y.append(value)

        # print(x[0:10])
        # dicoReviewLength={}
        # for i in TabReviewLength:
        #     if not i in dicoReviewLength:
        #         dicoReviewLength[i]=1
        #     else:
        #         dicoReviewLength[i]+=1




        # print(len(x))


        valx=x
        values = y
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
        # print(r_value)
        fig, ax = plt.subplots()
        plt.bar(valx, values)
        plt.suptitle('product review length')
        plt.ylabel('Number of reviews')
        plt.xlabel('review length')

        # plt.show()

csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def open_with_pandas_read_csv(filename):
    return pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)


def open_with_pandas_read_csv2(filename, delimiter):
    return   pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)
