import numpy as np
import scipy.stats

class EM :


    def __init__(self, nr_of_classes) :
        self.__nr_of_classes=nr_of_classes


    def fit(self, data):
        shape=data.shape
        nr_samples=shape[0]
        nr_features=shape[1]
        self.__nr_features=nr_features
        labels=np.random.randint(self.__nr_of_classes, size=nr_samples)
        pi=np.zeros(self.__nr_of_classes)
        mu=np.zeros((self.__nr_of_classes, nr_features))
        cov=np.zeros((self.__nr_of_classes, nr_features, nr_features))
        for k in range(self.__nr_of_classes):
            class_data=data[np.where(k==labels)]
            class_len=len(class_data)
            pi[k]=class_len/nr_samples
            variances=np.zeros(nr_features)
            for j in range(nr_features):
                feature_samples=[x[j] for x in class_data]
                mu[k][j]=np.mean(feature_samples)
                variances[j]=np.std(feature_samples)**2
            cov[k]=np.diag(variances)
        cont=True
        while cont:
            r_ik=np.zeros((nr_samples, self.__nr_of_classes))
            for i in range(nr_samples):
                x=data[i]
                P=np.zeros(self.__nr_of_classes)
                for k in range(self.__nr_of_classes):
                    P[k]=pi[k]*scipy.stats.multivariate_normal(mu[k],cov[k] + 0.01*np.eye(nr_features), allow_singular=True).pdf(x)
                for k in range(self.__nr_of_classes):
                    r_ik[i][k]=P[k]/sum(P)
            r=np.zeros(self.__nr_of_classes)
            L2=np.zeros(self.__nr_of_classes)
            for k in range(self.__nr_of_classes):
                r[k]=sum([x[k] for x in r_ik])
                pi[k]=r[k]/nr_samples
                mu_k=np.zeros(nr_features)
                for i in range(nr_samples):
                    mu_k=np.add(mu_k, r_ik[i][k]*data[i]/r[k])
                mu_old=mu[k]
                L2[k]= np.linalg.norm(np.add(mu_old,-mu_k))
                mu[k]=mu_k
                variances=np.zeros(nr_features)
                for i in range(nr_samples):
                    for j in range(nr_features):
                        if i==0:
                            variances[j]+=r_ik[i][k]/r[k]*data[i][j]**2-mu_k[j]**2
                        else:
                            variances[j]+=r_ik[i][k]/r[k]*data[i][j]**2

                cov[k]=np.diag(variances)
            if all(x<0.001 for x in L2):
                cont=False
        self.__pi=pi
        self.__mu=mu
        self.__cov=cov
            
    def predict(self, data):
        predicted=[]
        for x in data:
            P=np.zeros(self.__nr_of_classes)
            for k in range(self.__nr_of_classes):
                P[k]=self.__pi[k]*scipy.stats.multivariate_normal(self.__mu[k],self.__cov[k]+ 0.01*np.eye(self.__nr_features), allow_singular=True).pdf(x)
            predicted.append(np.argmax(P))
        return predicted