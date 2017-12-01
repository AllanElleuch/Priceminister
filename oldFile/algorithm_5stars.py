import pandas
import csv
import random
csv_delimiter = ';'


# filename : name of the csv file
def open_with_pandas_read_csv(filename):
    df = pandas.read_csv(filename, sep=csv_delimiter)
    data = df.values
    return data

def print_data(data):
    print(data[0])
    print(data[1])
    print(data[2])
    print(data[3])

dataset = open_with_pandas_read_csv("input_test.csv")
# frame = pandas.DataFrame.from_csv("input_test.csv", sep=csv_delimiter, encoding='utf-8-sig')
# print(frame["review_content"])
print_data(dataset[0])

# for x in range(80001,116395):
#     print(x)

with open('./output/algorithm_5stars.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["ID;Target"])
    for data in dataset:
        if(data[3]==5):
            writer.writerow([`int(data[0])`+";"+`1`])
        else:
            writer.writerow([`int(data[0])`+";"+`0`])




    # for x in range(80001,116395+1):
    #     writer.writerow([`x`+";"+`random.randint(0, 1)`])



# print(open_with_pandas_read_csv("input_train.csv")[0][0])
# print(open_with_pandas_read_csv("input_train.csv")[0][1])
# print(open_with_pandas_read_csv("input_train.csv")[0][2])
# print(open_with_pandas_read_csv("input_train.csv")[0][3])
