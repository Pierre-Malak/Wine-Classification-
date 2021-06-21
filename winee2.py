from __future__ import print_function
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
 # For data manipulation and analaysis.
# For data multidimentional collections and mathematical operations.
# For statistics Plotting Purpose
import matplotlib.pyplot as plt

# For Classification Purpose

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np, random, scipy.stats as ss
import sklearn.preprocessing

from sklearn import linear_model

from genetic_selection import GeneticSelectionCV
    

choice = input("press 1 for KNN\npress 2 for decision tree \npress 3 for K-Means CLustring \npress 4 for Genetic Algorithm")
if (choice == '1'):
    data = pd.read_csv('wine.csv') 
    
    
    def majority_vote_fast(votes):
        mode, count = ss.mstats.mode(votes)
        return mode
    
    def distance(p1, p2):
        return np.sqrt(np.sum(np.power(p2 - p1, 2)))
    
    def find_nearest_neighbors(p, points, k=5):
        distances = np.zeros(points.shape[0])
        for i in range(len(distances)):
            distances[i] = distance(p, points[i])
        ind = np.argsort(distances)
        return ind[:k]
    
    def knn_predict(p, points, outcomes, k=5):
        ind = find_nearest_neighbors(p, points, k)
        return majority_vote_fast(outcomes[ind])[0]
    def accuracy(predictions, outcomes):
        return 100*np.mean(predictions == outcomes)
    
    data.head()
    x=len(data)
    print("\n")
    
    print ("The Number of instances in the dataset is : " + str(x))
    print("\n")
    print(data.head()) #print dataset data
    
    data["is_red"] = (data["color"] == "red").astype(int)
    numeric_data = data.drop("color", axis=1)
    
    numeric_data.groupby('is_red').count()
    scaled_data = sklearn.preprocessing.scale(numeric_data)
    
    numeric_data = pd.DataFrame(scaled_data, columns = numeric_data.columns)
    print(numeric_data.head())
    
    knn = KNeighborsClassifier(81)  #K=sqrt(n)
    knn.fit(numeric_data, data['high_quality'])
    library_predictions = knn.predict(numeric_data)
    print("\n")
    print("Accuracy :")
    print(accuracy(library_predictions, data["high_quality"]))
    
    n_rows = data.shape[0]
    random.seed(123)
    selection=random.sample(range(n_rows), 10)
    selection
    predictors = np.array(numeric_data)
    training_indices = [i for i in range(len(predictors)) if i not in selection]
    outcomes = np.array(data["high_quality"])

elif (choice =='2'):
    dataset = pd.read_csv('wine.csv')

# Preprocessing Phase

# Checking having missing values
    print(dataset['high_quality'].value_counts())

# Replace missing values (NaN) with bulbbous stalk roots
    dataset['high_quality'].replace(np.nan, '1', inplace = True)

# Encoding textual values: Converting lingustic values to numerical values
    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(dataset.columns)):
        dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)
    
# Separating class color from the dataset features 
    X = dataset.drop('color', axis=1)
    y = dataset['color']

# Splitting dataset to training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle =True, test_size=0.3, random_state=42)
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train,y_train)
    predDTC = DTC.predict(X_test)
    reportDTC = classification_report(y_test,predDTC, output_dict = True)
    crDTC = pd.DataFrame(reportDTC).transpose()
    print(crDTC)

# Tree Visualisation

    fig = plt.figure(figsize=(100,80))
    plot = plot_tree(DTC, feature_names=list(dataset.columns), class_names=['RED', 'White'],filled=True)
    for i in plot:
        arrow = i.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor('black')
            arrow.set_linewidth(2)



elif (choice =='3'):
    df = pd.read_csv('wine.csv')

    #Make a copy of DF
    df_tr = df
    
    #Transsform the data to dummies
    df_tr = pd.get_dummies(df_tr, columns=['quality'])
    
    #Standardize
    clmns = ['fixed acidity', 'density','pH', 'alcohol','high_quality']
    df_tr_std = stats.zscore(df_tr[clmns])
    
    #Cluster the data
    kmeans = KMeans(n_clusters=2, random_state=42).fit(df_tr_std)
    labels = kmeans.labels_
    
    #Glue back to originaal data
    df_tr['cluster'] = labels
    
    #Add the column into our list
    clmns.extend(['cluster'])
    
    #Lets analyze the clusters
    
    print (df_tr[clmns].groupby(['cluster']).mean())
    
    
    
    print('Mean Accuracy :')
    print(np.mean(df_tr['high_quality']))
    
    

    
    
    #Scatter plot of Wattage and Duration
    sns.lmplot('chlorides', 'fixed acidity', data=df_tr, fit_reg=False, hue="color", scatter_kws={"marker": "D", "s": 100})
    plt.title('Chlorides vs fixed acidity')
    plt.xlabel('Chlorides')
    plt.ylabel('Fixed acidity')
    
    
elif (choice =='4'):
    gen = pd.read_csv('wine.csv')
    
    mappings = list()
    encoder =LabelEncoder()
    for column in range(len(gen.columns)):
        gen[gen.columns[column]] = encoder.fit_transform(gen[gen.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)


    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(gen), 20))
   # z = gen.drop('color', axis=1)
    X = np.hstack((gen, E))
    
    y = gen['color']

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)

     