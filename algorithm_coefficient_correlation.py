import pandas
import csv
import random
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from tools import toolKit

myTool = toolKit();
myTool.addDataframe('challenge_output_data_training_file_prediction_of_products_reviews_interests.csv')
myTool.addDataframe('input_train.csv')
# myTool.drawGraphReviewstars()
myTool.drawGraphReviewlength()
myTool.entrainerModelRandomForest()


def open_with_pandas_read_csv(filename):
    return pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)


def open_with_pandas_read_csv2(filename, delimiter):
    return   pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)

# dataFrameOutputTrain = open_with_pandas_read_csv()
# reviewInput = open_with_pandas_read_csv2('input_train.csv')
# tableauInputOutput=pandas.merge(reviewInput, reviewOutput)

csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))




 # Format d'une donnée : ID;review_content;review_title;review_stars;product

def print_data(data):
    print(data[0][0])
    print(data[0][1])
    print(data[0][2])
    print(data[0][3])
    print(data[0][4])



# dataset = open_with_pandas_read_csv2('input_train.csv',["review_stars","review_content"])
dataFrameInputTrain = open_with_pandas_read_csv2('input_train.csv',["review_stars"])
dataFrameOutputTrain = open_with_pandas_read_csv('challenge_output_data_training_file_prediction_of_products_reviews_interests.csv')
# dataFrame = open_with_pandas_read_csv2('challenge_output_data_training_file_prediction_of_products_reviews_interests.csv',["review_stars","review_content"])
# print(dataFrameOutputTrain.values)

# récupérer uniquement les commentaires utiles

# necessaire pour construire les graphs
tableauReviewStars = dataFrameInputTrain['review_stars'].values.tolist()
# y = dataFrameOutputTrain['Target'].values.tolist()


x = dataFrameInputTrain['review_stars'].values.tolist()
y = dataFrameOutputTrain['Target'].values.tolist()
# print(x[0:10000])
# print(y[0:10000])
# print(numpy.corrcoef([x[0:-1],y[0:-1]])[1,0])
# print(numpy.corrcoef([1,0,],[1,0])[1,0])

# print(numpy.corrcoef([[0,1],[1,2]]))
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
# print(r_value)
# // faire un graph en bar rating/ helpfulness
# // Faire un graph répartition length review / helpfulness
# // Faire un random forest sur ces deux paramètre


# premier graph !

#
#




####deuxième graphe en fonction des review length

def graphReviewlength():
    dataFrameInputReviewContent = open_with_pandas_read_csv2('input_train.csv',["review_content"])
    TabReviewLength =[]

    for review in dataFrameInputReviewContent.values.tolist():
        TabReviewLength.append(len(review[0]))

    dicoReviewLength={}
    for i in TabReviewLength:
        if not i in dicoReviewLength:
            dicoReviewLength[i]=1
        else:
            dicoReviewLength[i]+=1
    x,y=[],[] #x = valeur len  y = occurence

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
# graphReviewlength() #-0.473273081838

#

dataFrameInputReviewContent = open_with_pandas_read_csv2('input_train.csv',["review_content"])
TabReviewLength =[]
reviewOutput=dataFrameOutputTrain #.values.tolist()
reviewInput = open_with_pandas_read_csv2('input_train.csv',["review_content","ID"])
tableauInputOutput=pandas.merge(reviewInput, reviewOutput)
def entrainerModelRandomForest():
    #création colone len
    for review in dataFrameInputReviewContent.values.tolist():
        TabReviewLength.append(len(review[0]))
    tableauInputOutput['len'] = pandas.Series(np.array(TabReviewLength), index=tableauInputOutput.index)

    ## dico pour ploter review length
    dicoReviewLength={}
    for i in TabReviewLength:
        if not i in dicoReviewLength:
            dicoReviewLength[i]=1
        else:
            dicoReviewLength[i]+=1
    x,y=[],[]

    for key,value in dicoReviewLength.items(): # x=
        x.append(key)
        y.append(value)


    regr = RandomForestRegressor(max_depth=2, random_state=0)



    a = np.arange(6).reshape((3, 2))
    b=np.array(tableauInputOutput['len']).reshape((-1,1))
    c=np.array(tableauInputOutput['Target']).reshape((-1,1))
    regr.fit(b, c)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
    return regr
# print(regr.feature_importances_)
# print(regr.predict(np.array([191]).reshape(-1,1)))

# for revLength in tableauInputOutput['len'].values.tolist():

def testLenEfficiency(regr):
    lenList= tableauInputOutput['len'].values.tolist()
    tabGuess =[]
    tabGuess=(regr.predict(np.array(lenList).reshape(-1,1)))
    resultatGuess=[]
    for estimation in tabGuess:
        if estimation>0.5:
            resultatGuess.append(1)
        else :
            resultatGuess.append(0)
    res =0#taux de guess correct
    tabOutput= tableauInputOutput['Target'].values.tolist()
    for i in range(len(resultatGuess)):
        if( resultatGuess[i]==tabOutput[i]):
            res+=1
    print(res/len(tabOutput))
# regr = entrainerModelRandomForest()
# testLenEfficiency(regr)

# FAIRE UN RATIO COM UTILE/Catégorie

# print(tabGuess)

# print(b)



def save():
    with open(os.path.join(__location__, '/output/algorithm_random.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["ID;Target"])

        for x in range(80001,116395+1):
            writer.writerow([(str(x)+";"+str(random.randint(0, 1)))])
            # print((str(x)+";"+str(random.randint(0, 1))));
        print("done")

# save()

def save_dataframe():
    frame.to_csv("./output/test.csv",sep=';',columns=["review_stars"])
    # print(frame.keys())
    # print(frame.index.tolist())
# print(frame.keys())




# print(frame[['id']])
# frame.to_csv("test.csv",sep=';',columns=["index"])

        # writer.writerow([`x`+";"+`random.randint(0, 1)`])



# print(open_with_pandas_read_csv("input_train.csv")[0][0])
# print(open_with_pandas_read_csv("input_train.csv")[0][1])
# print(open_with_pandas_read_csv("input_train.csv")[0][2])
# print(open_with_pandas_read_csv("input_train.csv")[0][3])
