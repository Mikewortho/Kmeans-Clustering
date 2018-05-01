"""
Created on Wed Apr 11 11:07:43 2018

@author: michaelworthington
"""

from __future__ import division
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
from matplotlib import pyplot as p

#reads in the data sets combines them and also assigns correct labels
#seperates them and stores in a dict for later usage.
def labelMaker(file1, file2, file3, file4):
    labels = {}
    label_index = 0
    with open(file1, "r") as f, open(file2, "r") as a, open(file3, "r") as c, open(file4, "r") as v:
        fruits = f.readlines()
        animals = a.readlines()
        countries = c.readlines()
        veggies = v.readlines()
    data = fruits + animals + countries + veggies
        
    for line in data:
            label= line.strip().split()[0]
            labels[label_index]= label
            label_index += 1
    
    for lab in range(0, 58):
        labels[lab] = 'fruits'
    for lab in range(58, 108):
        labels[lab] = 'animals'
    for lab in range(108, 269):
       labels[lab] = 'countries'
    for lab in range(269, 329):
        labels[lab] = 'veggies'        
        
    return labels

#reads in the data sets combines them and also splits off the features 
#stores features in np.array 
def featureMaker(file1, file2, file3, file4):
    
    data = []
    with open(file1, "r") as f, open(file2, "r") as a, open(file3, "r") as c, open(file4, "r") as v:
        fruits = f.readlines()
        animals = a.readlines()
        countries = c.readlines()
        veggies = v.readlines()
    dataSet = fruits + animals + countries + veggies
    
    for line in dataSet:  
        tempData = []
        temp = line.strip().split(" ")
        
        for i in range(1, 301):
            tempData.append(temp[i])
        data.append(tempData)
  
    return np.array(data , dtype=float)               


# l2 normalisation of the feature array - dimensions 300 x 329
def l2feats(fVects):
    
    length = len(fVects)

    for i in range(length):
        
        norm = np.sqrt(np.sum(fVects[i] * fVects[i]))
        fVects[i] = fVects[i] / norm
        
    return fVects


#clustering method - uses the mean as the centroid
def clustering(k, data):

    #creating k initial centroids
    centroids = {}
    for i in range(k):
        centroids[i] = random.choice(data)

    
    for itera in range(500):
        #dict storing vectors representing valus (all clusters)
        clusterFeats = {}
        #dict storing indices representing values (all clusters)
        clustersIndices = {}

        for i in range(k):
            #lists storing vectors representing values
            clusterFeats[i] = []
            #lists storing indices of values in each cluster
            clustersIndices[i] = []

        for index, feat in enumerate(data):
            
            #euclidean distance
            distances = [np.sqrt(np.sum((feat - centroids[centroid])**2)) for centroid in centroids]
           
            #manhattan distance
            #distances = [np.sum(np.abs(feat - centroids[centroid])) for centroid in centroids]
            
            #cosine similarity 
            #distances = [np.dot(feat, centroids[centroid]) / (np.linalg.norm(feat) * np.linalg.norm(centroids[centroid]))for centroid in centroids]
            
           
            
            # ARGMAX for cosine only!!!!!
            #classification = np.argmax(distances)        

            # Comment out the line below in order to use cosine similarity measures  
            # MIN distances for euclidean and manhattan only!!!!
            classification = distances.index(min(distances))
            clusterFeats[classification].append(feat)
            clustersIndices[classification].append(index)

        #dictionary which contains originally generated centroids
        previous = dict(centroids)

        #calculating cluster average/mean - thus becoming new centroids
        for feat in clusterFeats:
            centroids[feat] = np.average(clusterFeats[feat], axis=0)


        optimized = True

        for c in centroids:
            original = previous[c]
            current = centroids[c]
            
            #Euclidian for centroid
            tolerance = np.sqrt(np.sum((current-original)**2))
            
            #Manhattan for centroid
            #tolerance = np.sum(np.abs(current-original))
            
            #cosine similarity for centroid
            #tolerance = np.dot(current, original) / (np.linalg.norm(current) * np.linalg.norm(original))     
            
            
            if tolerance > 0.0001:
                optimized = False

        if optimized:
            print ('\nConverged in %d iterations.' % (itera+1))
            break
    

    return clustersIndices



#labelling all elements of the cluster returning them as indices
def labelMe(clustersIndices, fLabels):

    ev = {}
    for label, feats in clustersIndices.items():
        
        temp = {}        
        for feat in feats:

            if fLabels[feat] not in temp:
                temp[fLabels[feat]] = 1                
            else:
                temp[fLabels[feat]] += 1
                
        ev[label] = temp
    
    return ev

#cleaning data adding 0's for empty clusters and outputting in matrix style format ready for eval
def matcher(dic, k):    
    classifications = np.zeros([k, 4])
    
    objects = ["fruits", "animals", "countries", "veggies"] 
    Cl = 0
    for i in dic:

        current_dic = dic[i]
        current_keys = current_dic.keys()
        
        obCount = 0
        for c in objects:
            if(c in current_keys):
                classifications[Cl][obCount] = current_dic[c]

            else:
                classifications[Cl][obCount] = 0
            obCount += 1
        Cl += 1 
        
    
    return classifications
    
#calculate pairs for evaluation
def pairs (n):
    return n*(n-1)/2


def total(sort):

    total = 0
    temp = 0

    for row in sort:
        temp = 0
        for j in row:  
            temp += j 
        
        total += pairs(temp)

    return total

#calcualate true positives
def TP(sort):
    
    TruePositives = 0
    for row in sort:
        for j in row:        
            TruePositives += pairs(j)

    return TruePositives

#calcualte false positives
def FP(total, TruePositives):

    FalsePositives = total - TruePositives
    
    return FalsePositives 



#calculate false negatives
def FN(sort, k):

    falseNegatives = 0
    for c in range (k):
        for i in range(3):
            if sort[c][i] == 0:
                continue
    
            for c2 in range(c+1, k):
                falseNegatives += sort[c][i] * sort[c2][i]
    
    return falseNegatives



# MAIN METHOD
if __name__ == '__main__':

    fr = 'fruits.txt'
    an = 'animals.txt'
    coun = 'countries.txt'
    vegs = 'veggies.txt'
    
    #Getting labels, all features, and feature vectors
    fLabels = labelMaker(fr,an,coun,vegs)
    fVects = featureMaker(fr,an,coun,vegs)

    #L2 normalisation of feature vetors and extracting arrays from the sparse matrix
    l2 = l2feats(fVects)
    
    
    #Creating lists to hold Marco-Averaged Recall, Precision and F-score for all cluster the range(2,21)
    macRecall = []
    macPrecision = []
    macF_Score = []
    
    r = range(1,11)
    for k in r:
        
        localRecall = 0
        localPrecision = 0
        localF_Score = 0
        
        print '\nClustering with K = %d' % (k)
        print "================================================================================"
        
       
        clustersIndices = clustering(k, l2)     
        #clustersIndices = clustering(k, fVects) 
        labelled = labelMe(clustersIndices, fLabels)
        sort = matcher(labelled, k)    
        
        
        print "================================================================================"
        totalItm = total(sort)
        TruePositives = TP(sort)
        print "True positives:", TruePositives       
        FalsePositives = FP(totalItm, TruePositives)
        print "False Positives:", FalsePositives 
        falseNegatives = FN(sort, k)     
        print "False Negatives:",falseNegatives
        localRecall = TruePositives / (TruePositives + falseNegatives)
        print "Local Recall:",localRecall 
        localPrecision = TruePositives / (TruePositives + FalsePositives)
        print "Local Precision:",localPrecision
        localF_Score = (2 * localRecall * localPrecision) / (localRecall + localPrecision)
        print "Local F_Score:",localF_Score
        
        macRecall.append(localRecall)        
        macPrecision.append(localPrecision)
        macF_Score.append(localF_Score)
        
    
    print "Macro-Averaged Recall:",sum(macRecall)/k
    print "Macro-Averaged Precision",sum(macPrecision)/k
    print "Macro-Averaged F_Score:",sum(macF_Score)/k
    
    p.subplot(1, 1, 1)
    p.plot(r, macRecall, '--y', label='Recall')
    p.plot(r, macPrecision, '--r', label='Precision')
    p.plot(r, macF_Score, '--b', label='F-Score')
    p.axis('tight')
    p.xticks(r)
    p.xlabel('Number of Clusters')
    p.ylabel('Macro-Averaged Measures %')
    p.legend(loc='upper center', prop={'size':6}, ncol=3)
    p.tight_layout()
    p.show()
