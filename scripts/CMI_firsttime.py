# coding=utf-8
import numpy as np
from multiprocessing import Pool
from mixedRVMI import CMIEstimate, estimateAllMI
import copy

def backwardFeatureSelection(threshold,features,target,res,k, nproc):
    'the function returns the selected features starting from the full dataset and removing features keeping the loss of information smaller than the threshold'

    featureScores= []
    relevantFeatures = features # at the beginning all features are included
    idMap = {k: k for k in range(relevantFeatures.shape[1])} # dictionary with original feature position
    CMIScore = 0 # cumulative loss of information
    sortedScores = []

    while CMIScore < threshold and relevantFeatures.shape[1]>1: 
        if nproc > 1:
            featureScores = scoreParallelFeatures(relevantFeatures, target, k, nproc)
        else: 
            featureScores = scoreFeatures(relevantFeatures, target, k) # for each feature it evaluates the I(Y,X_i|X_A), at first step I(Y,X_i|X_{-i}),...
        
        sortedScores = sorted(featureScores, key=lambda x:x[1]) # lista ordinata (ascending) in base al punteggio di ogni feature
        CMIScore += max(sortedScores[0][1],0) # se il punteggio più basso è negativo, prendo 0
        if CMIScore >= threshold: break
        relevantFeatures = np.delete(relevantFeatures, sortedScores[0][0], axis=1) # tolgo la feature (column) con punteggio più basso 
        #print("Removing original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
    res["numSelected"].append(relevantFeatures.shape[1]) 
    return list(idMap.values()) 

#def forwardFeatureSelection(threshold,features,target,res,k, nproc):
def forwardFeatureSelection(fold, features, target, k, CMI_threshold, CMI_n_features=-1, time=None, use_coordinates=False, starting_features=None, nproc=1, verbose=True):
    'returns the relevant features for the target starting from an empty array and populating it with the features that have the best CMI score'

    featureScores=[]
    idMap = {k: k for k in range(features.shape[1])} # dictionary with original feature position (positions in the remainingFeatures variable)
    idSelected = []
    selectedFeatures = [] # empty array at the beginning
    CMIScore = 0 
    remainingFeatures = copy.deepcopy(features)

    if starting_features is not None:
        remainingFeatures = np.delete(remainingFeatures, starting_features, axis=1)
        idSelected = starting_features
        for i in starting_features:
            selectedFeatures.append(features[:, i])

        count = 0
        idRemaining = [i for i in range(len(features)) if i not in idSelected]
        for featureID in idRemaining:
            idMap[count] = featureID
            count += 1
        
        print(f'Initial selected features for fold {fold}: {idSelected}', flush=True)

    elif time is not None:
        if time == 'year':
            num_constant_feature = 1
        elif time == 'month':
            num_constant_feature = 2
        elif time == 'month_year':
            num_constant_feature = 3

        if use_coordinates:
            num_constant_feature += 3

        for _ in range(num_constant_feature):
            selected = remainingFeatures.shape[1] - 1
            selectedFeatures.append(remainingFeatures[:, -1]) # append the best scoring feature to result
            idSelected.append(idMap[selected]) # save in a list of selected features ID    

            for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
                if a >= selected:
                    idMap[a] = idMap[a+1]
            idMap.pop(max(idMap))

            remainingFeatures = np.delete(remainingFeatures, selected, axis=1) # now score the remaining features
    else:
        firstBest = sorted(estimateAllMI(features, target, k), key=lambda x:x[1], reverse=True) # ordered list (descending) of features MI scores
        if verbose:
            print(firstBest)
            print("----- MI Scores -----", flush=True)
            print("Best MI score: {0}".format(firstBest[0][1]), flush=True)
            print("Adding first best original feature: {0}".format(idMap[firstBest[0][0]]), flush=True) # original feature position
        selectedFeatures.append(features[:, firstBest[0][0]]) # append the best scoring feature to result
        idSelected.append(idMap[firstBest[0][0]]) # save in a list of selected features ID
        
        CMIScore += firstBest[0][1]

        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= firstBest[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))

        remainingFeatures = np.delete(features, firstBest[0][0], axis=1) # now score the remaining features

    #while CMIScore < threshold and np.array(selectedFeatures).T.shape[1] < features.shape[1]:
    while np.array(selectedFeatures).T.shape[1] < features.shape[1]:
        if nproc > 1:
            featureScores = scoreParallelFeatures(remainingFeatures, target, k, nproc, np.array(selectedFeatures).T)
        else: 
            featureScores = scoreFeatures(remainingFeatures, target, k, np.array(selectedFeatures).T) 

        sortedScores = sorted(featureScores, key=lambda x:x[1], reverse=True) # scores in descending order
        #print(sortedScores)
        CMIScore += max(sortedScores[0][1], 0)
        if verbose:
            print("Highest CMI score: {0}".format(sortedScores[0][1]), flush=True)

        #if CMIScore >= threshold or sortedScores[0][1] <= 0: break # stop execution even if all scores are negative
        #if sortedScores[0][1] <= 0: break
        if CMI_n_features != -1:
            if len(selectedFeatures) == CMI_n_features: break
        else:
            if sortedScores[0][1] <= CMI_threshold: break

        selectedFeatures.append(features[:, idMap[sortedScores[0][0]]]) # select highest scoring feature
        remainingFeatures = np.delete(remainingFeatures, sortedScores[0][0], axis=1) # best scoring no longer needed for evaluation

        if verbose:
            print("Adding original feature: {0}".format(idMap[sortedScores[0][0]]), flush=True) # original feature position
        idSelected.append(idMap[sortedScores[0][0]])
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
        
        print(f'Selected features for fold {fold}: {idSelected}', flush=True)

    return idSelected

def scoreParallelFeatures(features, target, k, nproc, selected=None):
    'Versione con parallelismo dello score'
    args=[]
    for i in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,i,axis=1)
        args.append((features[:, i], target, selected, k))
    with Pool(nproc) as p:
        scores = p.starmap(CMIEstimate, args)
    scores = np.array(scores)
    return list(zip(range(len(scores)),scores))

def scoreFeatures(features, target, k, selected=None):
    'Ritorna una lista di features ID + punteggio CMI sul dato target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,col,axis=1)
        scores[col] = CMIEstimate(features[:, col], target, selected, k)
        #if scores[col] > 0 : 
            #print("CMI: {0}".format(scores[col]))

    return list(zip(range(len(scores)),scores))

def getThreshold(task, target, delta):
    if task == 1: # classification task
        return (delta**2)/2
    else:
        return delta/2*np.max(target)**2 # l-infinity norm
