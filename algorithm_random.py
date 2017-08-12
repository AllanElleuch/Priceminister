import pandas
import csv
import random
csv_delimiter = ';'

def open_with_pandas_read_csv(filename):
    df = pandas.read_csv(filename, sep=csv_delimiter)
    data = df.values
    return data

def print_data(data):
    print(data[0])
    print(data[1])
    print(data[2])
    print(data[3])

# dataset = open_with_pandas_read_csv("input_train.csv")
frame = pandas.DataFrame.from_csv("input_test.csv", sep=csv_delimiter, encoding='utf-8-sig')
frame2 = pandas.DataFrame.from_csv("challenge_output_data_training_file_prediction_of_products_reviews_interests.csv", sep=csv_delimiter, encoding='utf-8-sig')
frame=frame.merge(frame2,left_index =True,right_index =True, how='outer');
# print(frame)

# print(frame.keys())
# print(frame2.keys())
# print(frame["review_content"])
# print_data(dataset[0])

# for x in range(80001,116395):
#     print(x)

def save():
    with open('./output/algorithm_random.csv', 'w') as f:
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
