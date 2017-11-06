from tqdm import tqdm as pb
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import scipy as sc
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler


#baseDIR = '/home/pataki/synapse/gitParkinson/' # base directory of the github repo

def mainDFCreator(disease): # read the test and train metaFiles together
    trainDF = pd.read_csv(baseDIR + 'metaDB/trainMetaDF.tsv', sep='\t')
    trainDF = trainDF[trainDF[disease] >= 0] # throw away NaN as those are not necessary for this disease
    print('Train shape:', trainDF.shape)
    
    testDF = pd.read_csv(baseDIR + 'metaDB/testMetaDF.tsv', sep='\t')
    testDF = testDF[testDF[disease] == 'Score']    
    print('Test shape: ', testDF.shape)
    
    mainDF = trainDF.append(testDF)
    print('Merged:     ', mainDF.shape)

    if('tremorScore' != disease):
        mainDF.pop('tremorScore');
    if('bradykinesiaScore' != disease):
        mainDF.pop('bradykinesiaScore');
    if('dyskinesiaScore' != disease):
        mainDF.pop('dyskinesiaScore');
        
    return(mainDF)

def mainDFtrimmer(inputDF, fileMinLen, plot): # throw away samples with too many NaNs
    xLen = [] # list stores the length of X values along the metaDF
    yLen = []
    zLen = []
    fileLen = []

    for i in range(len(inputDF)):
        tmpFileName = inputDF.fileName.tolist()[i]
        tmpDF = pd.read_csv(tmpFileName, sep='\t')

        lX = sum([float(j)>-100.0 for j in tmpDF[tmpDF.columns[1]]]) # length of X column (without NaNs and values<=-100)
        lY = sum([float(j)>-100.0 for j in tmpDF[tmpDF.columns[2]]]) # length of Y column (without NaNs and values<=-100)
        lZ = sum([float(j)>-100.0 for j in tmpDF[tmpDF.columns[3]]]) # length of Z column (without NaNs and values<=-100)

        xLen.append(lX)
        yLen.append(lY)
        zLen.append(lZ)
        fileLen.append(len(tmpDF)) # length of the file with NaNs

    inputDF['xLen'] = xLen
    inputDF['yLen'] = yLen
    inputDF['zLen'] = zLen
    inputDF['fileLen'] = fileLen

    inputDF = inputDF[inputDF.xLen > fileMinLen] # drop too short files
    print('Remained shape:', inputDF.shape)
    
    if(plot): # plot histogram of lengths in different coordinates
        plt.rcParams['figure.figsize']=(15,2)
        plt.xlabel('Length of X column')
        plt.ylabel('count')
        inputDF.xLen.hist()
        plt.show()
        plt.xlabel('Length of Y column')
        plt.ylabel('count')
        inputDF.yLen.hist()
        plt.show()
        plt.xlabel('Length of Z column')
        plt.ylabel('count')
        inputDF.zLen.hist()
        plt.show()
    
    return(inputDF) # returns trimmed dataframe

colMap = {'GENEActiv_X':'X', 'Pebble_X':'X', 'GENEActiv_Y':'Y', 
          'Pebble_Y':'Y', 'GENEActiv_Z':'Z', 'Pebble_Z':'Z',
          'Pebble_Magnitude':'magnitude', 'GENEActiv_Magnitude':'magnitude'} # consistent column naming

def plotByID(ID, disease, inputDF, plotXmin = None, plotXmax = None): # plot the X-Y-Z curves for a certain ID-disease pair
    fN = inputDF[inputDF.dataFileHandleId == ID].fileName.tolist()[0] # file's path
    tmpDF = pd.read_csv(fN, sep='\t').rename(columns=colMap).astype('float').dropna(how = 'any', axis = 0)
    tmpDF['timestamp'] = tmpDF.timestamp - tmpDF.timestamp.tolist()[0] # scale time back to zero
    plt.plot(tmpDF.X)
    plt.plot(tmpDF.Y)
    plt.plot(tmpDF.Z)
    plt.legend()
    plt.title(str(ID) + ' - ' + str(disease) + ':' + str(inputDF[inputDF.dataFileHandleId == ID][disease].tolist()[0]))
    if((plotXmin != None) & (plotXmax != None)):
        plt.xlim(plotXmin, plotXmax)
    plt.show()
    
def dataByID(ID, inputDF): # return a pandas DF with the X-Y-Z curve data for the selected IDs
    fN = inputDF[inputDF.dataFileHandleId == ID].fileName.tolist()[0] # file's path
    tmpDF = pd.read_csv(fN, sep='\t').rename(columns=colMap).astype('float').dropna(how = 'any', axis = 0)
    tmpDF['timestamp'] = tmpDF.timestamp - tmpDF.timestamp.tolist()[0] # scale time back to zero
    return(tmpDF)

def FFTbyID(ID, cord, inputDF, trimFrom = 0, trimTo = -1): # return the Fourier power spectrum of a coordinate for an ID
    fN = inputDF[inputDF.dataFileHandleId == ID].fileName.tolist()[0] # file's path
    tmpDF = pd.read_csv(fN, sep='\t').rename(columns=colMap).astype('float').dropna(how = 'any', axis = 0)
    tmpDF['timestamp'] = tmpDF.timestamp - tmpDF.timestamp.tolist()[0] # scale time back to zero
    return(np.absolute(np.fft.rfft(tmpDF[cord].tolist()[trimFrom:trimTo]))**2)
    
def IDdesc(IDlist, inputDF): # return a DF about the metadata of an IDlist
    tmpDF = inputDF[inputDF.dataFileHandleId.isin(IDlist)]
    tmpDF.pop('fileName');
    return(tmpDF.set_index('dataFileHandleId'))

def featureBinarizer(inputDF, featureNames): # creates 0-1 features
    for col in featureNames:
        print(col, pd.unique(inputDF[col])) # all values for the given feature
        tmpBin = LabelBinarizer().fit_transform(inputDF[col].tolist())
        for i in range(len(tmpBin[0])):
            tmpCol = []
            for j in range(len(tmpBin)):
                tmpCol.append(tmpBin[j][i])
            inputDF[col + '_' + str(i+1)] = tmpCol
        inputDF.pop(col)
    return(inputDF)
