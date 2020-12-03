from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
#from sklearn.naive_bayes import CategoricalNB
import MNIST
import NCC
import NBC
import GNB


def main() :
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')


    train_features, test_features, train_labels, test_labels = mnist.get_data()

    mnist.visualize_random()
    
    ### GNB ###
    #gnb = GaussianNB()
    gnb=GNB.GNB()
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)
    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
    
    
    
    
    ### NCC ###
    
    #ncc_sk = NearestCentroid()
    #ncc = NCC.NCC()
    #ncc_sk.fit(train_features, train_labels)
    #ncc.fit(train_features, train_labels)
    #y_pred=ncc.predict(test_features)
    #y_pred_sk=ncc_sk.predict(test_features)



    #print("Classification report SKLearn NCC:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred_sk)))
    #print("Confusion matrix SKLearn NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred_sk))
    
    #print("Classification report my NCC:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred)))
    #print("Confusion matrix my NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
    
    
    
    ### NBC ###
    #nbc_sk = CategoricalNB()
    #nbc = NBC.NBC()
    #nbc_sk.fit(train_features, train_labels)
    #nbc.fit(train_features, train_labels)
    #y_pred=nbc.predict(test_features)
    #y_pred_sk=ncc_sk.predict(test_features)

    #print("Classification report SKLearn NBC:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred_sk)))
    #print("Confusion matrix SKLearn NBC:\n%s" % metrics.confusion_matrix(test_labels, y_pred_sk))
    
    #print("Classification report my NBC:\n%s\n"
    #  % (metrics.classification_report(test_labels, y_pred)))
    #print("Confusion matrix my NBC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    mnist.visualize_wrong_class(y_pred, 8)
    
    

if __name__ == "__main__": main()