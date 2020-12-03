import numpy as np
import scipy.stats
from time import sleep
from tqdm import tqdm

class GNB :


    def __init__(self) :
        self.__distributions = dict()
        self.__label_prob=dict()


    def fit(self, data, targets):
        feature_len=len(data[0])
        for label in set(targets):
            nr_samples=len(np.where(targets==label)[0])
            self.__label_prob[label]=float(nr_samples)/len(data)
            label_data=data[np.where(targets==label)]
            feature_dicts=[None]*feature_len
            for i in range(feature_len):
                feature_dicts[i]=dict()
                samples =[x[i] for x in label_data]
                feature_dicts[i]['distr']=scipy.stats.norm(np.mean(samples), np.std(samples)+0.01)
            
            self.__distributions[label]=feature_dicts


    def predict(self, data) :
        predicted = list()
        
        for features in tqdm(data):
            max_prob=0
            for label in self.__distributions:
                prospect_prob=self.__label_prob[label]
                for i, feature in enumerate(self.__distributions[label]):
                    prospect_prob*=feature['distr'].pdf(features[i])
                if prospect_prob >= max_prob:
                    max_prob = prospect_prob
                    predicted_label=label
            predicted.append(predicted_label)
        return predicted