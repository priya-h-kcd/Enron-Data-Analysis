#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot
sys.path.append("../tools/")
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #SVC or SVM?
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from tester import test_classifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

poi = ["poi"]

financial_features =  ['salary', 'deferral_payments', 
					'total_payments', 'loan_advances', 
					'bonus', 'restricted_stock_deferred', 
					'deferred_income', 'total_stock_value', 
					'expenses', 
					'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features =  ['to_messages', 'email_address', 
					'from_poi_to_this_person', 
					'from_messages', 
					'from_this_person_to_poi', 
					'shared_receipt_with_poi']


email_features.remove('email_address')



 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print("Total data point in dataset :", len(data_dict))



### Task 2: Remove outliers

## Checking the outliers based on scatterplot between salary and bonus
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
salarylst =[]
bonuslst =[]
for item in data:
	salarylst.append(item[0])
	bonuslst.append(item[1])
salarylst = sorted(salarylst)
bonuslst = sorted(bonuslst)
for key in data_dict:
	if data_dict[key]['salary'] == salarylst[-1]:
		print(key, data_dict[key]['salary'] )
	elif data_dict[key]['salary'] == salarylst[-2]:
		print(key, data_dict[key]['salary'])

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### The plot and data shows that the data point showing salary around 26 million is the total of all the salaries.
###Removing the key 'TOTAL' from dictionary.
###Another highest salary is of SKILLING JEFFREY K who is one of poi
data_dict.pop('TOTAL')	
data = featureFormat(data_dict, features)






for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()




## Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict



def computeFraction( num, den ):
 	fraction = 0.
 	if num == "NaN" or den == "NaN":
 		fraction = 0.
 	else:
 		fraction = float(num)/float(den)
 	return fraction

def add_new_feature(dict, num, den, new_feature):
 	numerator = dict[num]
 	denominator = dict[den]
 	dict[new_feature] = computeFraction(numerator,denominator)
 	return dict




for data_point in my_dataset:
	# Fraction of emails from poi to total emails recieved 
	my_dataset[data_point] = add_new_feature(my_dataset[data_point],
												'from_poi_to_this_person', 
												'to_messages', 
												'fraction_from_poi_total')

	#Fraction of emails to poi to total emails sent
	my_dataset[data_point] = add_new_feature(my_dataset[data_point],
												'from_this_person_to_poi', 
												'from_messages', 
												'fraction_to_poi_total')

	#Fraction of exercised stock options to total stock value
	my_dataset[data_point] = add_new_feature(my_dataset[data_point],
												'exercised_stock_options', 
												'total_stock_value', 
												'fraction_exercised_stock_total')

    
email_features = email_features + ['fraction_from_poi_total', 'fraction_to_poi_total']  
financial_features = financial_features + ['fraction_exercised_stock_total']

features_list = poi + financial_features + email_features
print("Total new features created :", 3)
print("Total  features used :", len(email_features)+ len(financial_features))
####################


### Extract features and labels from dataset for local testing


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "length of data numpy array:", len(data)
# print "Features List:", features_list
# # print "Labels:", labels
# # print "Features:", features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
algo_list = [ 'Decision_Tree', 'Naive_Bayes', 'SVM', 'K_Nearest_Neighbors', 'Adaboost', 'Random_Forest']

def create_clf(algorithm):
	params = {}
	if algorithm == 'Decision_Tree':
		clf = DecisionTreeClassifier()
		params = { algorithm + "__min_samples_split": [30, 35, 40]}

	elif algorithm == 'Naive_Bayes':
		clf = GaussianNB()
	
	elif algorithm == 'SVM':
		clf = SVC()
		params = {algorithm + '__kernel' : ['rbf'], algorithm + '__C': [1000, 10000]}

	elif algorithm == 'K_Nearest_Neighbors':
		clf = KNeighborsClassifier()
		params = {algorithm + '__n_neighbors' : [3, 5, 8]}

	elif algorithm == 'Adaboost':
		clf= AdaBoostClassifier()
		params = {algorithm + '__n_estimators' : [5, 10, 20, 30]}

	elif algorithm == 'Random_Forest':
		clf = RandomForestClassifier()
		params = {algorithm + "__n_estimators":[3, 4, 5,6]}

	return clf, params


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


for algo in algo_list:
    clf, clf_params = create_clf(algo)
    print algo
    
    scaler = MinMaxScaler()
    scaled_feat = scaler.fit_transform(features_train)
    # print "Scaled Values:\n:", scaled_feat
    
    
                                            
    pipe = Pipeline(steps=[
                            ('FeatureScaling', MinMaxScaler()),
                            ('SelectBestFeature', SelectKBest()),
                            (algo, clf)
                          ]
                    )
    
    param = {'SelectBestFeature__k' : [5, 6, 7, 8, 9, 10, 11, 12],
              'SelectBestFeature__score_func' : [f_classif]
             }
    param.update(clf_params)
    
    sss = StratifiedShuffleSplit(labels_train, n_iter = 20, test_size = 0.5,
                                 random_state = 0)
                                 
                 
    gs = GridSearchCV(pipe,
                       param,
                       verbose = 0,
                       scoring = 'f1_weighted',
                       cv=sss
                       )
    
    gs.fit(features_train, labels_train)
    
    pred = gs.predict(features_test)

    clf = gs.best_estimator_

     ## Printing selected features
    pipe.fit(features_train, labels_train)
    selected_features = gs.best_estimator_.named_steps['SelectBestFeature'].get_support(indices=True)
    feature_scores = gs.best_estimator_.named_steps['SelectBestFeature'].scores_
    sfs = []
    for sf in selected_features: 
        sfs.append((features_list[sf + 1], feature_scores[sf]))         
    print len(sfs), "best parameters with scores:"
    for f, s in sfs: print f, "{0:.3f}".format(s)


  
    print("\n", algo, "performance report:")
    print(classification_report(labels_test, pred))

    #WTesting algorithms against  test_classifier...'
    print '\Testing algorithms against  test_classifier...'
    #print "\nFeatures_list for test_classifier:", features_list
    test_score = test_classifier(clf, my_dataset, features_list)
    print test_score 
    
  	


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)