from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def main():
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


if __name__ == "__main__":
    main()