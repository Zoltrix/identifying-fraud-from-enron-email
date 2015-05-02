#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


def compute_fraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.

    fraction = poi_messages / all_messages

    return fraction

# ## Task 1: Select what features you'll use.
# ## features_list is a list of strings, each of which is a feature name.
# ## The first feature must be "poi".

# You will need to use more features
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees']

my_feature_list = features_list+['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                                 'shared_receipt_with_poi', 'fraction_to_poi']


# Use these lists to test the decision tree classifer
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
#my_feature_list = features_list + ['fraction_to_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for name in my_dataset:
    data_point = my_dataset[name]

    ## create a new feature 'fraction_to_poi' and 'fraction_from_poi'
    ## indicating the fraction of emails sent to and from poi's
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest

## print the scores of the features in descending order
selector = SelectKBest(k=5).fit(features, labels)
print sorted(zip(my_feature_list[1:], selector.scores_), key=lambda tup: tup[1], reverse=True)

#clf = DecisionTreeClassifier(min_samples_split=11, random_state=42)
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
test_classifier(clf, my_dataset, my_feature_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, my_feature_list)