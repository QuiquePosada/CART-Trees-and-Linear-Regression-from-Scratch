"""
Author  : Enrique Posada Lozano
ID      : A01700711
ITESM Campus Qro
Artificial Intelligence Project
"""

"""
The following project employs the use of machine learning techniques made both from scratch and with the use of the scikit framework 
in order to compare and contrast the performance in predictions between a logistic regression and a random forest, and also makes use 
of a linear regression and a k-means
"""

# Import required libraries
import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split # For splitting dataset data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plot
import time
# import seaborn

# import operator # helps in obtaining minimum and maximum in a list of tuples

# Classification libraries
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree

# Linear regression library
from sklearn.linear_model import LinearRegression

# Global variables
totalErrorLinearRegression = []

# --------- Decision Tree Functions

def getMSE(y, mean):
    """
        Calculates the MSE (Mean Square Error) of the data given
        This is the impurity function used for regression trees

    """
    # print("\nMY Y\n",mean)
    if len(y) == 0: # Done in order to avoid obtaining nan values if there are no elements contained in y
        return 0
    mse = numpy.average((y-mean)**2)
    # print("mse\t",mse)
    return mse

# Entropy/Gini are the Cost Functions for Decision Trees (Classification)
def getEntropy(data):
    """
        Calculates the Entropy for a given labeled Data
        This is the impurity function for classification trees

        data    -> Data provided for labels
        It is used to obtain unique elements the make up a class and applies Shannon's entropy formula

        Formula is as follows
        np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    """
    _, elementsCounted = numpy.unique(data, return_counts = True) # Obtains data only for the indices provided using loc
    totalElements = numpy.sum(elementsCounted)
    acc = (elementsCounted/totalElements) * numpy.log2(elementsCounted/totalElements) # numpy makes this type of operation faster without the need of a for loop    
    acc = - numpy.sum(acc) # axis=0 for rows
    # USED After a a new split is made
    return acc

def informationGain(systemEntropy, label, leftPartitionIndices, rightPartitionIndices):
    """
        Measures the quality of a split
        Acts as a reduction in noise

        systemEntropy           -> The entropy of the current Node in tree
        label                   -> The class values of the whole training dataset
        leftPartitionIndices    -> The obtained left indices obtained from the split
        rightPartitionIndices   -> The obtained right indices obtained from the split
    """

    leftLabels = label.loc[leftPartitionIndices]
    rightLabels = label.loc[rightPartitionIndices]
    # info gain 
    # fi = Frequency of label at left branch
    leftFi = len(leftPartitionIndices) / (len(leftPartitionIndices) + len(rightPartitionIndices))
        # Calculate the entropy for the quality of the Split
    entropy = (leftFi * getEntropy(leftLabels)) + ((1 - leftFi) * getEntropy(rightLabels))
    # impurity = (leftFi * getEntropy(leftLabels)) + ((1 - leftFi) * getEntropy(rightLabels)))
    infoGain = systemEntropy - (entropy)
    # infoGain = systemEntropy - (impurity)
    joinedElementIndices = leftPartitionIndices.union(rightPartitionIndices)
    joinedElements = numpy.unique(label.loc[joinedElementIndices], return_counts = True)
    return infoGain, joinedElements

def splitData(data, column, value):
    """
        Splits the data given into left and right branches by comparing data greater than a given value and data less than given value
        Update, obtains only indices in order to be faster

        data    -> Dataset
        column  -> Current feature being evaluated
        value   -> Value provided to evaluate the dataset and return the indices of those
    """
    left = data[data[column] <= value]
    right = data[data[column] > value]
    return left.index,right.index

def findBestSplit(data, label, type):
    """
        Calculates and gets the Best Split for the tree
        returns a dictionary containing the best split found

        data    -> The data from which it should find the best split
        label   -> The class values for given data
        type    -> "regression" or "classification"
    """

    # variables containing information 
    bestSplit = {}

    # Check for the best feature to make the split
    if type == "classification":
        systemEntropy = getEntropy(label) # Calculate the entropy for the system
        bestSplit = { "bestValue": None, "bestFeature": None, "entropy": systemEntropy, "bestInformationGain": 0, "bestClassValues":None, "bestLeftChildren": None, "bestRightChildren": None }
        
        # Original Implementation
        for feature in data:
            # print("Current Feature: ",feature)
            elements, _ = numpy.unique(data[feature], return_counts = True)
            # print("Unique elements\t",elements)
            # print(elementsCounted)
            for value in elements:
                leftPartitionIndices, rightPartitionIndices = splitData(data, feature, value) # Just like the Yes/No Excel Worksheet
                infoGain, classValues = informationGain(systemEntropy, label, leftPartitionIndices, rightPartitionIndices)
                if infoGain > bestSplit["bestInformationGain"]:
                    print("BestGain\t"+str(infoGain),end="\r")
                    bestSplit["bestFeature"] = feature
                    bestSplit["bestValue"] = value
                    bestSplit["bestInformationGain"] = infoGain
                    bestSplit["bestClassValues"] = classValues
                    # This saves the two branches made
                    bestSplit["bestLeftChildren"] = leftPartitionIndices
                    bestSplit["bestRightChildren"] = rightPartitionIndices
    else:
        bestSplit = { "bestFeature": None, "mse": 0, "value":0, "bestMSE": float('inf'), "bestLeftChildren": None, "bestRightChildren": None }

        bestSplit["value"] = numpy.average(label)

        for feature in data:
            elements, _ = numpy.unique(data[feature], return_counts = True)
            # print("CURRENT FEATURE\t",feature)
            for val in elements:
                leftPartitionIndices, rightPartitionIndices = splitData(data, feature, val) # Just like the Yes/No Excel Worksheet
                mseLeftRight = getMSE(label.loc[leftPartitionIndices.union(rightPartitionIndices)],numpy.average(label))
                leftMean = numpy.average(label.loc[leftPartitionIndices]) if len(leftPartitionIndices) > 0 else 0
                rightMean = numpy.average(label.loc[rightPartitionIndices]) if len(rightPartitionIndices) > 0 else 0

                leftFi = len(leftPartitionIndices) / (len(leftPartitionIndices) + len(rightPartitionIndices))

                # Calculate the quality of the Split
                branchesMSE = (leftFi * getMSE(label.loc[leftPartitionIndices],leftMean)) + ((1 - leftFi) * getMSE(label.loc[rightPartitionIndices],rightMean))
                # Condition for establishing the best split (the one with least MSE)
                if branchesMSE < bestSplit["bestMSE"]:
                    print("BEST MSE\t"+str(branchesMSE.round(5))+" and Label Value "+str(bestSplit["value"]), end="\r")
                    # print("Feature best\t",feature)
                    bestSplit["bestFeature"] = feature
                    bestSplit["bestValue"] = val
                    # bestSplit["value"] = numpy.average(label)
                    # bestSplit["bestMSE"] = branchesMSE
                    bestSplit["mse"] = mseLeftRight 
                    bestSplit["bestMSE"] = branchesMSE
                    # bestSplit["bestClassValues"] = classValues
                    # This saves the two branches made
                    bestSplit["bestLeftChildren"] = leftPartitionIndices
                    bestSplit["bestRightChildren"] = rightPartitionIndices
    return bestSplit


# NOTE: The decision tree implementation is based on 
    # https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ and https://towardsdatascience.com/algorithms-from-scratch-decision-tree-1898d37b02e0
    # Visit for more information as this shows one how to build a decision tree from scratch
    # https://www.python-course.eu/Decision_Trees.php
def insertLeaf(samples,label):
    """
        Inserts a leaf node which consists of 
        samples ->  Data contained for the amount of elements that were counted for a given class/label
        labels  ->  Data contained for the class
    """
    # print("A leaf has been added")
    labelValue = pandas.unique(label)
    return { "samples":len(samples), "class":labelValue[0], "elementsCounted":len(samples), "isLeaf": True }

def buildDecisionTreeClassifier(samples, y, max_depth, currentLevel=0):
    """
    Builds a Decision Tree for Classification
    If the best information gain is 0, then it inserts a leaf node, else it branches the tree out recursively 
    """
    # print("current Level!\t",currentLevel)
    bestSplit = findBestSplit(samples,y,"classification")
    # print("\n")
    # print("Split\n"+str(bestSplit),end="\r")
    # If best gain is equal to 0, then it is a leaf node :)
    if bestSplit["bestInformationGain"] == 0 or currentLevel == max_depth:
        return insertLeaf(samples,y)
    # print("\n")
    # if not, creates more branches for the tree (Recursively)
    leftBranch = buildDecisionTreeClassifier(samples.loc[bestSplit["bestLeftChildren"]], y.loc[bestSplit["bestLeftChildren"]], max_depth, currentLevel+1)
    rightBranch = buildDecisionTreeClassifier(samples.loc[bestSplit["bestRightChildren"]], y.loc[bestSplit["bestRightChildren"]], max_depth, currentLevel+1)
    return { "feature": bestSplit["bestFeature"], "value": bestSplit["bestValue"], "infoGain": bestSplit["bestInformationGain"], "classValues": bestSplit["bestClassValues"], "leftBranch": leftBranch, "rightBranch": rightBranch, "isLeaf": False,  }

def buildDecisionTreeRegression(samples, y, max_depth, currentLevel=0):
    """
    Builds the Decision Tree
    Makes use of MSE 
    If the impurity is 0, then it inserts a leaf node, else it branches the tree out recursively 
    """
    # print("Current Level!\t",currentLevel,end="\r")
    # print("\n")
    bestSplit = findBestSplit(samples,y,"regression")
    # print("BEST SPLIT GREAT! ", bestSplit)
    # print("\n")
    # print("Current is TRUE?\t"+str(currentLevel >= max_depth)+"\n")
    # HERE YOU ALSO CHECK and declare the greatest level for insight purposes

    # If best gain is equal to 0, then it is a leaf node :)
    if bestSplit["mse"] == 0 or currentLevel == max_depth:
        return insertLeaf(samples,y)
    # if not, creates more branches for the tree (Recursively)
    leftBranch = buildDecisionTreeRegression(samples.loc[bestSplit["bestLeftChildren"]], y.loc[bestSplit["bestLeftChildren"]], max_depth, currentLevel+1)
    rightBranch = buildDecisionTreeRegression(samples.loc[bestSplit["bestRightChildren"]], y.loc[bestSplit["bestRightChildren"]], max_depth, currentLevel+1)
    return { "feature": bestSplit["bestFeature"], "value": bestSplit["bestValue"], "valAvg": bestSplit["value"], "mse": bestSplit["mse"], "bestMSE": bestSplit["bestMSE"],  "leftBranch": leftBranch, "rightBranch": rightBranch, "isLeaf": False,  }


def checkPredictions(instance, tree):
    """
        Traverses a given tree until a leaf is reached, returning the value to predict from a given instance
        instance    -> pandas tuple containing instance data
        tree        -> the decisionTree previously trained
    """
    # Check if it has currently reached a leaf node, and return the predicted label
    if tree["isLeaf"] == True:
        # print("CLASS IS!!!!\t", tree["class"])
        return tree["class"]
    
    # traverse left branch of tree
    # feature = tree["feature"]
    if instance[tree["feature"]] <= tree["value"]:
        return checkPredictions(instance, tree["leftBranch"])
    else:
        return checkPredictions(instance, tree["rightBranch"])
    return 1

def predict(samples, tree):
    """
        Predicts unseen data
        This is used for validating and testing the tree built
        This function is recursive since it consists on traversing the tree until a leaf node is reached, returning the class it belongs to

        data    -> Test Data or data from which we simply want to predict
        tree    -> The tree created to make tests on Data
    """
    # samples.apply(checkPredictions, axis=1,)
    predictions = list()
    for instance in range(len(samples)):
        pred = checkPredictions(samples.iloc[instance], tree)
        predictions.append(pred)
    # print("predictions\t",predictions)
    # print("MY TREE\t",tree)
    # print("Done")
    return predictions

def decisionTree(samples, y, type, max_depth=None):
    """
        This is the starting point to creating a tree and training the model
        samples -> Training Features
        y       -> Training Labels
        type    -> "regression" or "classification" 
    """
    tree = {}
    print("MAX DEPTH ",max_depth)
    if max_depth != None and max_depth <= 0:
        print("\n\tError in Tree : max depth should be greater than 0")
        exit()
    if type == "classification":
        tree = buildDecisionTreeClassifier(samples, y, max_depth)
        return tree
    tree = buildDecisionTreeRegression(samples, y, max_depth)
    return tree

# ---- end ----

# ------------------------- Random Forest Implementation
def baggingPredictions(samples, forest):
    """ 
        Aggregates predictions and returns a list of the predicted values      
        samples -> The sample of test features/instamces to predict
        forest  -> The forest created and to be used to predict
    """
    predictions = list()
    # Get all predictions in tree
    for tree in forest:
        predictions.append(predict(samples, tree))
    # Aggregate predictions
    predictions = numpy.array(predictions)
    forestPredictions = list()
    for row in predictions.T.tolist(): # transposes array and makes it a list so you have all predictions from each tree in a row
        forestPredictions.append(max(set(row),key=row.count))
    return forestPredictions
    # return predictions

def randomForest(samples, y, numTrees, type, max_depth=None):
    """
        # This forest implementation uses bagging and feature randomness when building each individual tree to try to create a forest of trees 
        (NOTE : The performance did not provide better results, therefore it was better to stick with a "smart" tree since it requires more trees than 10 to work )
        
        # In theory, it is better to have many trees to guarantee it works than one tree that "does" it all 
        # While some trees may be wrong, many other trees will be right

        samples     -> Training Features (Pandas DataFrame)
        y           -> Training Labels (Pandas DataFrame))
        numTrees    -> number of Trees to create the forest
        type        -> "classification" or "regression"
    """
    nRows = samples.shape[0] # Use all rows
    nFeatures = int(numpy.sqrt(samples.shape[1])) # Truncates float and converts it to an integer
    trees = list()
    for _ in range(numTrees):
        # SciKit considers the number of features to choose to be the Square Root of the column features
        sample = samples.sample(n=nRows,replace=False).sample(n=nFeatures,replace=False,axis=1) # replace True makes it endless to reach the next branch
        print("sample\n",sample)
        print("Y\n",y.loc[sample.index])
        # myTree = tree.DecisionTreeRegressor(criterion="mse")
        # myTree.fit(sample,y.loc[sample.index])
        myTree = decisionTree(sample,y.loc[sample.index],type,max_depth)
        print("TREE DONE!!!")
        trees.append(myTree)
    return trees


# ------------------------- Linear Regression Functions 

def calculateHyp(params, sample):
    """
        Calculates the predicted value (hypothesis)
        yHat = a+bx1+cx2+ ... nxn   # Simplified Formula for Linear Regression

        yHat -> Predicted Value,    b0 -> Bias,     b1 -> Coefficient of 1st parameter ,    bn -> n coefficent 
        x1 -> first input (feature -> ex: Age value),  xn -> n input/feature 

        params -> Coefficientes of each parameter
        sample -> instance contained in the features dataset
    """
    acc = 0
    acc = params * sample # pandas takes care of multiplying each parameter with the respective feature
    # Optimized version
    acc = acc.sum(axis=1) # To sum all in row and not columns, axis is set to 1
    # print(acc)
    # exit()
    # acc = acc * (-1) # This is done for logistic regression
    return acc

def gradientDescent(params, samples, labels, learningRate):
    """
        Applies gradient descent in order to obtain new and "better" coefficient params (parameters)
        error = predictedValue - expectedValue

        params          ->
        samples         ->
        learningRate    ->
        labels          ->
    """
    error = 0
    newParams = list(params)

    # Optimized Version
    # acc = 0
    yHat = calculateHyp(params,samples)
    error = (yHat - labels)
    acc = numpy.dot(error,samples)
    newParams = params - (learningRate * ((1 / len(samples)) * acc)) # GD which in turn calculate the new params using alpha (learning rate)
    # print("newP",newParams)
    return newParams

def showErrors(params, samples, labels):
    """
    Calculates the error (Model Performance) (based on Benji's implementation)

    params  -> Coefficientes of each parameter
    samples -> All the training data
    y       -> All the real output data
    """
    # global totalError
    # error_acum = 0
    # error = 0

    # Optimized version
    hyp = calculateHyp(params,samples)
    # error = numpy.vectorize(crossEntropy)(hyp,y)
    # error = numpy.vectorize()

    mean_error = numpy.sqrt(getMSE(labels,hyp)) # cost function 

    # print("MSE\t",mean_error)
    # test = (labels - hyp)**2
    # print("\nTest\n",test)
    # print("ANSWER\t",test.sum()/(len(samples)))
    # error_sum = error.sum() # REMOVED since getMSE already does this

    # PREVIOUS Implementation
    # for instance in range(len(samples)):
    #     hyp = calculateHyp(params,samples.iloc[instance])
    #     error = crossEntropy(hyp, y.iloc[instance])
    #     error_acum = error_acum + error # this error is different from the one used to update, this is general for each sentence it is not for each individual param
    # mean_error = error/len(samples)  # Removed since getMSE already does this
    totalErrorLinearRegression.append(mean_error) # Append to list showing error progress
    # print("DONE!!!")
    # exit()
    return mean_error

class CustomLinearRegression():
    """
        Custom Linear Regression
        Needs to be initiated with a given sample (x) and label (y), alongside a learning rate (alpha)
    """
    def __init__(self, samples, labels, alpha):
        self.labels = labels
        self.alpha = alpha # Learning Rate
        # Starting Coefficients/Weights (For each parameter)
        params = numpy.zeros(samples.shape[1]).tolist()
        # params = numpy.zeros(samples.shape[1])
        # Add a new column and param for the Bias
        # params.append(0) # Bias coefficient
        # Establish parameters starting value with random values
        params = numpy.random.rand(len(params))
        # self.samples = samples.assign(Bias=1)
        self.samples = samples
        self.params = params # Coefficients
        self.predictedValues = list() # list of predicted values

    def train(self):
        # Current epoch iteration
        epoch = 0 
        start_time = time.time() # Keep track of current time passed
        # predicted_Values = []
        params = self.params
        # error = 0

        # While loop that stops until local minimum is reached or there is no further improvement in the bias
        while True:
            prevParams = list(params) # previous epoch coefficients
            params = gradientDescent(params,self.samples,self.labels, self.alpha)
            error = showErrors(params, self.samples, self.labels) # calculates the error between predicted and real data
            params = list(params) # In order to leave in same format as before -> not in a numpy array
            if(params == prevParams or epoch == 5000 or error < 5): # the loop will only end if no further changes are made/seen in the params, the number of epochs given is reached or a given minimum error is reached
                yHat = calculateHyp(params,self.samples)
                yHat = yHat.to_numpy()
                # predicted_Values = yHat
                self.predictedValues = yHat
                self.params = params
                # print("predicted values")
                # print(predicted_Values)
                    # print("Expected -> %.3f , Predicted Value -> %.3f [%d]" % (trainingLabel.iloc[instance], yhat, round(yhat)))
                print ("FINAL params :")
                print (params)
                print("THE TRAINING HAS FINISHED IN " + str(epoch) + " EPOCHS!!")
                finishedTrainingTime = time.time() - start_time
                print("The training lasted for " + str(finishedTrainingTime/60) + " minutes")
                break
            epoch += 1
            print("EPOCHS -> " + str(epoch) + " and error -> " + str(error), end="\r") # Overwrites the current line
        print("Training finished")
# ---- end ----

def scaleData(features):
    """
    Usually done for regressions
    Normalizes features in order for gradient descent to work correctly (improves the convergence speed of the logistic regression algorithm)
    features is an arg containing the sample of feature data to be normalized

    Normalization is made using Rescaling (min-max normalization) (https://en.wikipedia.org/wiki/Feature_scaling)

    returns the features of the dataset in a normalized manner and ( the minValues and maxValues for scaling data when a new query is made )

    normalizedVal = (x - min(x)) / (max(x) - min(x)))

    features -> The features to be normalized in the dataset
    """
    # print("\nMAX OF DATASET")
    # print(features.max())
    maxValues = features.max()
    # print("\nMIN OF DATASET")
    # print(features.min())
    minValues = features.min()
    print("\nInitializing Normalization ...")
    scaledFeatures = (features - minValues) / (maxValues - minValues)
    print("Finished Normalization\n")
    return scaledFeatures, minValues, maxValues


# Custom Query
def getCommonInput(dataframe):
    """
        Obtains the common input for making queries once models are trained

        Since what is chosen ranges from 0 to N, some options add a certain number in order to fit the original criteria of the dataset
            (Ex : Never, Sometimes and Always questions are added 1 since the dataset ranges the answer from 1 to 3)

        returns a pandas dataframe for use
    """
    dataframe["Gender"] = [int(input("\nAre you Male or Female?\n\t0 - Female\n\t1 - Male\n"))]
    dataframe["Age"] = [float(input("\nWhat is your Age?\n"))]
    dataframe["Height"] = [float(input("\nWhat is your Height?\n"))]
    dataframe["family_history_with_overweight"] = [int(input("\nHave you had past family members with obesity/overweight problems?\n\t0 - No\n\t1 - Yes\n"))]
    dataframe["FAVC"] = [int(input("\nDo you frequently eat high caloric food?\n\t0 - No\n\t1 - Yes\n"))]
    dataframe["FCVC"] = [1 + int(input("\nHow often do you eat vegetables with your meals?\n\t0 - Never\n\t1 - Sometimes\n\t2 - Always\n"))]
    dataframe["NCP"] = [1 + int(input("\nHow many main meals do you eat on a daily basis?\n\t0 - Between 1 and 2\n\t1 - 3 (Normal Quantity)\n\t2 - More than three meals\n"))]
    dataframe["SMOKE"] = [int(input("\nDo you smoke?\n\t0 - No\n\t1 - Yes\n"))]
    dataframe["TUE"] = [int(input("\nHow much time do you spend on electronic devices (videogames, computers, tv, phone) in a daily basis?\n\t0 - 0-2 hours\n\t1 - 3-5 hours\n\t2 - More than 5 hours\n"))]
    dataframe["FAF"] = [int(input("\nHow often do you exercise (do physical activity) on a weekly basis?\n\t0 - I do not exercise\n\t1 - 1-2 days\n\t2 - 2-4 days\n\t3 - 4-5 days (or more...)\n"))]
    dataframe["CH2O"] = [1 + int(input("\nHow much water do you drink daily?\n\t0 - Less than 1 liter\n\t1 - Between 1 and 2 liters\n\t2 - More than 2 liters\n"))]
    dataframe["SCC"] = [int(input("\nDo you monitor the amount of calories you eat on a daily basis?\n\t0 - No\n\t1 - Yes\n"))]

    # Multiple values in column - REQUIRE TO SEPARATE AS IN ONE HOT ENCODING
    _caec = int(input("\nDo you eat any food between meals?\n\t0 - No\n\t1 - Sometimes\n\t2 - Frequently\n\t3 - Always\n"))
    _calc = int(input("\nHow often do you drink alcohol?\n\t0 - I don't drink\n\t1 - Sometimes\n\t2 - Frequently\n\t3 - Always\n"))
    _mtrans = int(input("\nWhich way of transportation do you use on a daily basis?\n\t0 - Car\n\t1 - Motorbike\n\t2 - Bike\n\t3 - Public Transportation\n\t4 - Walking\n"))

    # Process columns that can have multiple values
    # caec
    dataframe['CAEC_Always'], dataframe['CAEC_Frequently'], dataframe['CAEC_Sometimes'], dataframe['CAEC_no'] = 0,0,0,0
    if _caec == 0:
        dataframe['CAEC_no'] = [1]
    elif _caec == 1:
        dataframe['CAEC_Sometimes'] = [1]
    elif _caec == 2:
        dataframe['CAEC_Frequently'] = [1]
    else:
        dataframe['CAEC_Always'] = [1]
    
    # calc
    dataframe['CALC_Always'], dataframe['CALC_Frequently'], dataframe['CALC_Sometimes'], dataframe['CALC_no'] = 0,0,0,0
    if _calc == 0:
        dataframe['CALC_no'] = [1]
    elif _calc == 1:
        dataframe['CALC_Sometimes'] = [1]
    elif _calc == 2:
        dataframe['CALC_Frequently'] = [1]
    else:
        dataframe['CALC_Always'] = [1]

    # mtrans
    dataframe['MTRANS_Automobile'], dataframe['MTRANS_Bike'], dataframe['MTRANS_Motorbike'], dataframe['MTRANS_Public_Transportation'], dataframe['MTRANS_Walking'] = 0,0,0,0,0
    if _mtrans == 0:
        dataframe["MTRANS_Automobile"] = [1]
    elif _mtrans == 1:
        dataframe["MTRANS_Motorbike"] = [1]
    elif _mtrans == 2:
        dataframe["MTRANS_Bike"] = [1]
    elif _mtrans == 3:
        dataframe["MTRANS_Walking"] = [1]
    elif _mtrans == 4:
        dataframe["MTRANS_Public_Transportation"] = [1]
    else:
        dataframe["MTRANS_Walking"] = [1]

    # At the end it returns a dataframe or numpy array ()
    return dataframe

def queryClassification():
    """
    Used for making classification queries using the models made. (Decision Tree by Hand and SCIKIT decision tree classifier implementation)
    Obtains several input for dataframe columns and calls each models predict function and shows the prediction from each one
    """
    print("Calculating Obesity Level with classification models\n")
    dataframe = getCommonInput(pandas.DataFrame({}, columns=classificationFeatures.columns))
    dataframe["Weight"] = float(input("\nWhat is your weight?\n"))
    print("Prediction Made")
    fwTreepred = treeClasif.predict(dataframe)
    if fwTreepred[0] == 0:
        fwTreepred = "Insufficient Weight"
    elif fwTreepred[0] == 1:
        fwTreepred = "Normal Weight"
    elif fwTreepred[0] == 2:
        fwTreepred = "Overweight Level I"
    elif fwTreepred[0] == 3:
        fwTreepred = "Overweight Level II"
    elif fwTreepred[0] == 4:
        fwTreepred = "Obesity Type I"
    elif fwTreepred[0] == 5:
        fwTreepred = "Obesity Type II"
    elif fwTreepred[0] == 6:
        fwTreepred = "Obesity Type III"

    treePred = predict(dataframe, classificationTree)
    if treePred[0] == 0:
        treePred = "Insufficient Weight"
    elif treePred[0] == 1:
        treePred = "Normal Weight"
    elif treePred[0] == 2:
        treePred = "Overweight Level I"
    elif treePred[0] == 3:
        treePred = "Overweight Level II"
    elif treePred[0] == 4:
        treePred = "Obesity Type I"
    elif treePred[0] == 5:
        treePred = "Obesity Type II"
    elif treePred[0] == 6:
        treePred = "Obesity Type III"
    return print("Predicted Classification Tree -> %s , Predicted SCIKIT Classification Tree -> %s" % (treePred, fwTreepred))

def queryRegression():
    """
    Used for making regression queries using the models made. (Linear Regression by Hand, SCIKIT Linear Regression implementation, Decision Tree by Hand and SCIKIT decision tree regressor implementation)
    Obtains several input for dataframe columns and calls each models predict function and shows the prediction from each one
    """
    print("Calculating weight with regression models\n")
    dataframe = getCommonInput(pandas.DataFrame({}, columns=regressionFeatures.columns))
    dataframe['NObeyesdad_Insufficient_Weight'], dataframe['NObeyesdad_Normal_Weight'], dataframe['NObeyesdad_Obesity_Type_I'], dataframe['NObeyesdad_Obesity_Type_II'], dataframe['NObeyesdad_Obesity_Type_III'], dataframe['NObeyesdad_Overweight_Level_I'], dataframe['NObeyesdad_Overweight_Level_II'] = 0,0,0,0,0,0,0
    _nobeyesdad = int(input("\nSelect a weight category to which the person relates to :\n\t0 - Insufficient Weight\n\t1 - Normal Weight\n\t2 - Overweight Type I\n\t3 - Overweight Type II\n\t4 - Obsesity Type I\n\t5 - Obsesity Type II\n\t6 - Obsesity Type III\n"))
    if _nobeyesdad == 0:
        dataframe['NObeyesdad_Insufficient_Weight'] = [1]
    elif _nobeyesdad == 1:
        dataframe['NObeyesdad_Normal_Weight'] = [1]
    elif _nobeyesdad == 2:
        dataframe['NObeyesdad_Overweight_Level_I'] = [1]
    elif _nobeyesdad == 3:
        dataframe['NObeyesdad_Overweight_Level_II'] = [1]
    elif _nobeyesdad == 4:
        dataframe['NObeyesdad_Obesity_Type_I'] = [1]
    elif _nobeyesdad == 5:
        dataframe['NObeyesdad_Obesity_Type_II'] = [1]
    else:
        dataframe['NObeyesdad_Obesity_Type_III'] = [1]

    # ScaleValues
    dataframe[['Age','Height','FCVC','NCP','CH2O','FAF','TUE']] = (dataframe[['Age','Height','FCVC','NCP','CH2O','FAF','TUE']] - regressionMinValues) / (regressionMaxValues - regressionMinValues)
    print("Prediction Made")
    fwTreepred = fwregressionTree.predict(dataframe)
    fwLRpred = lin.predict(dataframe)
    lrPred = calculateHyp(regressionModel.params, dataframe)
    treePred = predict(dataframe, regressionTree)    
    return print("Predicted LR Value -> %.3f , Predicted SCIKIT LR Value -> %.3f , Predicted Regression Tree -> %.3f , Predicted SCIKIT Regression Tree -> %.3f" % (lrPred[0], fwLRpred[0], treePred[0], fwTreepred[0]))


# ---- end ----









# Obtain data 
csvFileName = "./ObesityDataSet_raw_and_data_sinthetic/ObesityDataSet_raw_and_data_sinthetic.csv"

dataset = pandas.read_csv(csvFileName)

""" 
STEP:
    Preprocess data
"""
# One of the most important use of feature engineering is that it reduces overfitting and improves the accuracy of a model.

# Classification Pre-Processing
    # Single value in column (Yes/No, Male/Female)
gender = pandas.get_dummies(dataset["Gender"]).drop("Female", axis=1) # Use Gender to convert to 1's or 0's and drop one of the columns since it is unnecesary
familyHistory = pandas.get_dummies(dataset["family_history_with_overweight"]).drop("no",axis=1) 
favc = pandas.get_dummies(dataset["FAVC"]).drop("no",axis=1)
smoke = pandas.get_dummies(dataset["SMOKE"]).drop("no",axis=1)
scc = pandas.get_dummies(dataset["SCC"]).drop("no",axis=1)
    # NOTE: Most of these drop an unnessary column since it is not really needed at the time of processing data 

    # Multiple values in column
caec = pandas.get_dummies(dataset["CAEC"], prefix="CAEC")
calc = pandas.get_dummies(dataset["CALC"], prefix="CALC")
mtrans = pandas.get_dummies(dataset["MTRANS"], prefix="MTRANS")

classificationDataset = dataset.drop(["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"], axis=1)
# Round continous values that do not require many unique values with decimal places (Ex: Age with decimals or Height with many decimals)
    # This does not help the values since the dataset was preprocessed before hand

# Add new encoded values
classificationDataset["Gender"], classificationDataset["family_history_with_overweight"], classificationDataset["FAVC"], classificationDataset["SMOKE"], classificationDataset["SCC"]  = [gender, familyHistory, favc, smoke, scc]
classificationDataset = pandas.concat([classificationDataset,caec,calc,mtrans], axis=1)

# Label handling
    # In this case, the label is treated with a label encoding technique, in order to avoid more features and make it one column for the model to understand
    # There are 7 types of Obesity, therefore labeling will be done from 0 to 6 for the model to understand
    # NOTE: In a Logistic Regression, one hit encoding is needed since the model needs to know for each one if the option is needed or not
# factorizedLabel = pandas.factorize(dataset["NObeyesdad"])
# classificationDataset["NObeyesdad"] = factorizedLabel[0]
factorizedLabel = (dataset["NObeyesdad"].replace({"Insufficient_Weight":0,"Normal_Weight":1,"Overweight_Level_I":2,"Overweight_Level_II":3,"Obesity_Type_I":4,"Obesity_Type_II":5,"Obesity_Type_III":6}).to_numpy(),pandas.Index(["Insufficient_Weight","Normal_Weight","Overweight_Level_I","Overweight_Level_II","Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]))
classificationDataset["NObeyesdad"] = factorizedLabel[0]
# print(factorizedLabel)

# NOTE: Now making them ordinal (with order) to check if there is a better performance from the model

# print(scaleData(forestDataset))
print("I think we finished checking the model preprocess data")

# Get Features and Labels for Classification
classificationFeatures = pandas.DataFrame(classificationDataset, columns = [
    'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC',
    'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
    'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
    'MTRANS_Public_Transportation', 'MTRANS_Walking'
])
classificationLabels = classificationDataset["NObeyesdad"]
# Split the Dataset into a training dataset and a test dataset
    # In this case, the model is trained with 75% (3/4 of data) of the data given 
        # (NOTE: Since no random_state is given, the seed by default is random for each time the code is executed)
classificationTrainingFeatures, classificationTestFeatures, classificationTrainingLabels, classificationTestLabels = train_test_split(classificationFeatures, classificationLabels, test_size=0.25)

# Regression Pre-Processing
nObeyesdad = pandas.get_dummies(dataset["NObeyesdad"], prefix="NObeyesdad") # Uses One Hot Encoding for the previous label/class 
# Drop original columns
regressionDataset = dataset.drop(["Gender", "Weight", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"], axis=1)
regressionDataset, regressionMinValues, regressionMaxValues = scaleData(regressionDataset) # scale data ("REALLY ESSENTIAL!!!!!!! since it brings weights infinity if not")
# Scale for regression

# Insert dataframe with single column with the new values
regressionDataset["Gender"], regressionDataset["family_history_with_overweight"], regressionDataset["FAVC"], regressionDataset["SMOKE"], regressionDataset["SCC"]  = [gender, familyHistory, favc, smoke, scc]
# Insert dataframes with mutliple columns
regressionDataset = pandas.concat([regressionDataset, caec, calc, mtrans, nObeyesdad], axis=1)
# Insertion of Bias (For this project it did not really help)
# regressionDataset["Bias"] = 1

regressionFeatures = pandas.DataFrame(regressionDataset, columns=[
    'Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 
    'SCC', 'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking' ,
    'NObeyesdad_Insufficient_Weight', 'NObeyesdad_Normal_Weight',
    'NObeyesdad_Obesity_Type_I', 'NObeyesdad_Obesity_Type_II',
    'NObeyesdad_Obesity_Type_III', 'NObeyesdad_Overweight_Level_I',
    'NObeyesdad_Overweight_Level_II'
])
regressionLabels = dataset["Weight"]


regressionTrainingFeatures, regressionTestFeatures, regressionTrainingLabel, regressionTestLabel = train_test_split(regressionFeatures, regressionLabels, test_size=0.25)

print("Finished pre-processing")


# queryClassification()
# queryRegression()

# ------------------------------------
# Decision Tree Classifier

realStart = time.time()

# SCIKIT
treeClasif = tree.DecisionTreeClassifier(criterion="entropy")
treeClasif.fit(classificationTrainingFeatures, classificationTrainingLabels)

# rf = randomForest(classificationTrainingFeatures, classificationTrainingLabels, 5, "classification")
# print(rf)

# Small version of the tree
fig, axes = plot.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300) 
tree.plot_tree(treeClasif,feature_names=classificationFeatures.columns,class_names=factorizedLabel[1],filled = True)
fig.savefig('clasifTreeSmall.png')

print("Visualizing tree")
# fig = plot.figure(figsize=(200,125)) # 250,200
# _ = tree.plot_tree(treeClasif, 
#                    feature_names=classificationFeatures.columns,  
#                    class_names=factorizedLabel[1],
#                    filled=True,
#                    fontsize=24)
# fig.savefig('proHighFinal.png')
y_predTree = treeClasif.predict(classificationTestFeatures)
cm = confusion_matrix(classificationTestLabels, y_predTree)
print(cm)
plot.matshow(cm)
plot.title('SCIKIT Estimation of "Obesity" Level')
plot.colorbar()
plot.ylabel('Real Value')
plot.xlabel('Predicted Value')
plot.show()

# My implementation of a decision tree
classificationTree = decisionTree(classificationTrainingFeatures, classificationTrainingLabels, "classification")
# print("Classfification Tree\n",classificationTree)

pred = predict(classificationTestFeatures, classificationTree)
print("SCIKIT Decision Tree Classifier Accuracy : "+str(accuracy_score(classificationTestLabels, y_predTree))+" %")
print("SCIKIT's model has " + str(cm.diagonal().sum()) + " correct predictions and " + str(classificationTestFeatures.shape[0]-cm.diagonal().sum()) + " are wrong out of " + str(classificationTestLabels.shape[0]))

cm = confusion_matrix(classificationTestLabels, pred)
print(cm)
correctPredictions = cm.diagonal().sum()
plot.matshow(cm)
plot.title('"Obesity" Level')
plot.colorbar()
plot.ylabel('Real Value')
plot.xlabel('Predicted Value')
plot.show()

print ("\nAccuracy of Decision Tree Model with Test Data (%) : ", accuracy_score(classificationTestLabels, pred)) 
print("Decision Tree has " + str(correctPredictions) + " correct predictions and " + str(classificationTestFeatures.shape[0]-correctPredictions) + " are wrong out of " + str(classificationTestLabels.shape[0]))


# Regression 

# Framework implementation of Linear Regression
lin = LinearRegression()
lin.fit(regressionTrainingFeatures, regressionTrainingLabel)

# # Make predictions with test features
lrpred = lin.predict(regressionTestFeatures)
print("RMSE SCIKIT Linear Regression : "+str(numpy.sqrt(mean_squared_error(regressionTestLabel, lrpred))))
print("R Squared SCIKIT Linear Regression :\t",r2_score(regressionTestLabel, lrpred)) # measures how close the data is to the fitted line as a percentage



# --------- Linear Regression ------------

# Learning Rate for GD
alpha = 0.305
totalErrorLinearRegression = []

regressionModel = CustomLinearRegression(regressionTrainingFeatures, regressionTrainingLabel, alpha)
regressionModel.train()
plot.plot(totalErrorLinearRegression)
plot.title("Error")
plot.xlabel("# Epochs")
plot.ylabel("Error (RMSE)")
plot.show()

yhat = calculateHyp(regressionModel.params,regressionTestFeatures)

print("RMSE Linear Regression : "+str(numpy.sqrt(mean_squared_error(regressionTestLabel, yhat))))
print("R Squared Linear Regression :\t",r2_score(regressionTestLabel, yhat)) # measures how close the data is to the fitted line as a percentage

# for instance in range(len(regressionTestFeatures)):
#     print("Expected -> %.3f , Predicted Value -> %.3f " % (regressionTestLabel.iloc[instance], yhat.iloc[instance]))


# Decision Tree Regressor
# Framework
fwregressionTree = tree.DecisionTreeRegressor(criterion="mse", max_depth=10)
fwregressionTree.fit(regressionTrainingFeatures, regressionTrainingLabel)
newpred = fwregressionTree.predict(regressionTestFeatures)
print("RMSE SCIKIT Decision Tree Regressor : "+str(numpy.sqrt(mean_squared_error(regressionTestLabel, newpred))))
print("R Squared SCIKIT Tree Regressor :\t",r2_score(regressionTestLabel, newpred)) # measures how close the data is to the fitted line as a percentage
print(fwregressionTree.get_depth())
print("Visualizing tree")
# Small version of tree
fig, axes = plot.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300) 
tree.plot_tree(fwregressionTree,
            feature_names = regressionFeatures.columns, 
            class_names=pandas.unique(regressionLabels),
            filled = True)
fig.savefig('regTreeSmall.png')

plot.show()

# fig = plot.figure(figsize=(200,125)) 
# _ = tree.plot_tree(regressionTree, 
#                    feature_names=regressionFeatures.columns,  
#                 #    class_names=pandas.unique(regressionLabels),
#                    filled=True,
#                    fontsize=24)
# fig.savefig('treeRegFinalDepth.png')


# Hand implementation
# Comparing both models
start = time.time()
regressionTree = decisionTree(regressionTrainingFeatures, regressionTrainingLabel, "regression", 10)
finishedTrainingTime = time.time() - start
print("\n\nThe training lasted for " + str(finishedTrainingTime/60) + " minutes")
print("regression Tree finished!!!!!!\n",regressionTree)
predRegTree = predict(regressionTestFeatures, regressionTree)
print("RMSE Regression Tree : "+str(numpy.sqrt(mean_squared_error(regressionTestLabel, predRegTree))))
print("R Squared Tree Regression :\t",r2_score(regressionTestLabel, predRegTree)) # measures how close the data is to the fitted line as a percentage

for instance in range(len(regressionTestFeatures)):
    print("Expected -> %.3f , Predicted LR Value -> %.3f , Predicted SCIKIT LR Value -> %.3f , Predicted Regression Tree -> %.3f , Predicted SCIKIT Regression Tree -> %.3f" % (regressionTestLabel.iloc[instance], yhat.iloc[instance], lrpred[instance], predRegTree[instance], newpred[instance]))


print("\nWhole program took "+str((time.time() - realStart)/60)+" minutes")

