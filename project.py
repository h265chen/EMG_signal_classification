import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from dataClass import lift_file
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

# read text data as dataFrame
def readtext(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'hand switch', 'box switch', 'MS', 'MS','MS','MS','MS','MS']
    return df


def plotcolumn(subject, weight, trail, column):
        # # subject = 1     # 1,2,3,4,5
        # weight = '2pt5kg'   # 0kg, 2pt5kg, 5kg, 10kg, 15kg, 20kg
        # trail = 1   # 1, 2, 3, 4, 5
    address = os.getcwd() + '\\Data' + '\\S0' + str(subject) + ' Raw Data 30 Files' + '\\LiftCycle_' + weight + str(trail) + '.txt'
    data = readtext(address)
    plt.subplot(2, 1, 1)
    plt.title('subject' + str(subject) + str(weight) + str(column) + str(trail))
    plt.plot(data[str(column)])
    return data


# calculate RMS for each column
def calRMS(subject, weight, trail, column, window_size, overlap):
    rms = list()
    ms = 0
    # get one column from plotText()
    for window_beg in range(0, 19800, overlap):
        for i in range(window_beg, window_size+window_beg):
            ms = ms + data.iloc[i][str(column)] ** 2
        ms = ms / window_size
        x = math.sqrt(ms)
        rms.append(x)
    plt.subplot(2, 1, 2)
    plt.title('RMS subject' + str(subject) + str(weight) + str(column)+str(trail))
    plt.plot(rms, color='r')
    print(rms)

def plotRMS(dataFrame):

    col_list = ['BB', 'TB', 'BR', 'AD', 'LES', 'TES']
    ms = 0
    window_beg = 0
    window_size = 200
    overlap = 10
    plot_index = 0
    for column_name in col_list:
        col = dataFrame[column_name]
        rms = list()
        for window_beg in range(0,len(col)-window_size,overlap):
            for i in range(window_beg, window_size+window_beg):
                ms = ms + col[i]**2
            ms = ms/window_size
            x = math.sqrt(ms)
            rms.append(x)
        plt.subplot(6, 1, plot_index + 1)
        plt.title(column_name)
        plt.ylabel('RMS value')
        plt.plot(rms)
        plot_index= plot_index +1
    plt.show()



def plotBoxData(df,df1,df2):
    plt.subplot(3,1,1)
    plt.title('Box Switch Subject 2 for trial 1 for 2.5 kg')
    plt.plot(df['box switch'])

    plt.subplot(3,1,2)
    plt.title('Box Switch Subject 2 for trial 1 for 10 kg')
    plt.plot(df1['box switch'])

    plt.subplot(3,1,3)
    plt.title('Box Switch Subject 2 for trial 1 for 20 kg')
    plt.plot(df2['box switch'])
    plt.tight_layout()
    plt.show()




def runNaiveBayes(dataframe):
    X = dataframe[['BB', 'TB', 'BR', 'AD', 'LES', 'TES']]
    Y = dataframe['weight']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))

    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def createDFObjects():
    print("Running classification")
    subject = 1
    trial = 1
    weight = '2pt5kg'
    address = os.getcwd() + '\\Data' + '\\S0' + str(subject) + ' Raw Data 30 Files' + '\\LiftCycle_' + weight + str(
        trial) + '.txt'
    lf = lift_file(address)
    df1 = lf.get_df()
    df1['weight'] = 0

    weight = '10kg'
    address = os.getcwd() + '\\Data' + '\\S0' + str(subject) + ' Raw Data 30 Files' + '\\LiftCycle_' + weight + str(
        trial) + '.txt'
    lf2 = lift_file(address)
    df2 = lf.get_df()
    df2['weight'] = 1

    weight = '20kg'
    address = os.getcwd() + '\\Data' + '\\S0' + str(subject) + ' Raw Data 30 Files' + '\\LiftCycle_' + weight + str(
        trial) + '.txt'
    lf3 = lift_file(address)
    df3 = lf3.get_df()
    df3['weight'] = 2

    frames = [df1, df2,df3]
    result = pd.concat(frames)
    return result


if __name__ == '__main__':

    df = createDFObjects()
    print(df)
    #df = readtext(address)
    #plotRMS(df)
    #plotRMS(df)
    #plotBoxData(df)
