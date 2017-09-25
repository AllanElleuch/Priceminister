import pandas
import csv
import random
import os
import scipy.stats
import numpy
import matplotlib.pyplot as plt

csv_delimiter = ';'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def open_with_pandas_read_csv(filename):
    df = pandas.read_csv(os.path.join(__location__, filename), sep=csv_delimiter)
    # data = df.values
    data = df
    return data


def open_with_pandas_read_csv2(filename, delimiter):
    df = pandas.read_csv(os.path.join(__location__, filename),sep=csv_delimiter,  usecols=delimiter)
    # print(df.loc[:,'ID':'review_content'])
    # print(df.loc[:, ['ID','review_content']])
    # print(df['review_content'])
    # maList = df['review_content']
    # maList.tolist()
    # print(maList[0])
    # print(df.loc[:,'ID':])
    # data = df.values
    return df


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


tableauReviewStars = dataFrameInputTrain['review_stars'].values.tolist()
y = dataFrameOutputTrain['Target'].values.tolist()


# // faire un graph en bar rating/ helpfulness
# // Faire un graph répartition length review / helpfulness
# // Faire un random forest sur ces deux paramètre


# premier graph !
def graphReviewstars:

    tableauRepartitionReviewStars=[0,0,0,0,0]
    for i in range(len(tableauReviewStars)):
        index = tableauReviewStars[i]
        print(index)
        tableauRepartitionReviewStars[index-1] +=1


    valx=[1,2,3,4,5]
    names = ['1 stars', '2 stars', '3 stars', '4 stars', '5 stars']
    values = tableauRepartitionReviewStars
    fig, ax = plt.subplots()
    plt.bar(valx, values)
    ax.set_xticks(valx)
    ax.set_xticklabels(names)
    plt.suptitle('product review stars')
    plt.show()
#
#


####deuxième graphe en fonction des review length

def graphReviewlength:
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
    x,y=[],[]

    for key,value in dicoReviewLength.items():
        x.append(key)
        y.append(value)
    print(dicoReviewLength)


    valx=x
    values = y
    fig, ax = plt.subplots()
    plt.bar(valx, values)
    plt.suptitle('product review length')
    plt.ylabel('Number of reviews')
    plt.xlabel('review length')

    plt.show()


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
