import numpy as np
from time import sleep
from copy import copy


class NBC :


    def __init__(self) :
        self.__probabillities = dict()
        self.__label_prob=dict()


    def fit(self, data, targets):
        feature_len=len(data[0])
        
        for label in set(targets):
            feature_dicts=[None]*feature_len
            
            nr_samples=len(np.where(targets==label)[0])
            self.__label_prob[label]=float(nr_samples)/len(data)
            for i in np.where(targets==label)[0]:
                for j in range(feature_len):
                    if not feature_dicts[j]:
                        feature_dicts[j]=dict()    
                    if data[i][j] in feature_dicts[j]:
                        feature_dicts[j][data[i][j]]+=1
                    else:
                        feature_dicts[j][data[i][j]]=1
            for j in range(feature_len):
                for value in feature_dicts[j]:
                    feature_dicts[j][value]=feature_dicts[j][value]/float(nr_samples)
                
            self.__probabillities[label]=feature_dicts
                    


    def predict(self, data) :
        predicted = list()
        for features in data:
            max_prob=0
            for label in self.__probabillities:
                prospect_prob=self.__label_prob[label]
                for i, feature_val in enumerate(features):
                    if feature_val in self.__probabillities[label][i]:
                        prospect_prob*=self.__probabillities[label][i][feature_val]
                    else:
                        prospect_prob*=0.0001
                if prospect_prob >= max_prob:
                    max_prob = prospect_prob
                    predicted_label=label
            predicted.append(predicted_label)
        return predicted