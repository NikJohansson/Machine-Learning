import numpy as np


class NCC :


    def __init__(self) :
        self.__centroids = dict()


    def fit(self, data, targets):
        feature_len=len(data[0])
        for label in set(targets):
            centroid=np.zeros(feature_len)
            for i in np.where(targets==label)[0]:
                for j in range(feature_len):
                    centroid[j]+=data[i][j]/feature_len
            self.__centroids[label]=centroid
                    


    def predict(self, data) :
        predicted = list()
        for features in data:
            min_dist=float("inf")
            for label in self.__centroids:
                centroid=self.__centroids[label]
                dist= np.linalg.norm(centroid-features)
                if dist< min_dist:
                    pred=label
                    min_dist=dist
            predicted.append(pred)
            
        return predicted