import pandas
import csv
import random
import os
import scipy.stats
import numpy

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


x = dataFrameInputTrain['review_stars'].values.tolist()
y = dataFrameOutputTrain['Target'].values.tolist()
print(x[0:3])
print(y[0:3])
print(numpy.corrcoef([x[0:-1],y[0:-1]])[1,0])

# print(numpy.corrcoef([[0,1],[1,2]]))
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)

# print_data(dataset);
# print(dataset);
# dataset = open_with_pandas_read_csv("input_test.csv")
# print(open_with_pandas_read_csv("input_test.csv","ID"))
#Pour une compatiblité entre window et linux on va chercher à calculer le chemin absolu pour accéder des fichiers


# frame = pandas.DataFrame.from_csv(os.path.join(__location__, 'input_test.csv'), sep=csv_delimiter, encoding='utf-8-sig')
# frame2 = pandas.DataFrame.from_csv(os.path.join(__location__, "challenge_output_data_training_file_prediction_of_products_reviews_interests.csv"), sep=csv_delimiter, encoding='utf-8-sig')
# frame=frame.merge(frame2,left_index =True,right_index =True, how='outer');
# print(frame)

# print(frame.keys())
# print(frame2.keys())
# print(frame["review_content"])
# print_data(dataset[0])

# for x in range(80001,116395):
#     print(x)

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
