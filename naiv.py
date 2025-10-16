import pandas as pd
from collections import Counter

def naive_bayes_classifier(X_train,y_train,X_test):
    class_counts = Counter(y_train)
    prior_probs = {label : count/len(y_train) for label,count in class_counts.items()}

    conditional_probs = {}
    for feature in X.columns:
        conditional_probs[feature] = {}
        for class_label in y_train.unique():
            feature_vlaues = X_train[y_train==class_label][feature]
            value_counts = Counter(feature_vlaues)
            total_count = len(feature_vlaues)

            conditional_probs[feature][class_label] = {label : count/total_count for label,count in value_counts.items()}

    final_probabilities = {}

    for class_label in y_train.unique():
        probability = prior_probs[class_label]
        for feature in conditional_probs:
            value = X_test[feature].iloc[0] 
            if value in conditional_probs[feature][class_label] :
                probability *= conditional_probs[feature][class_label][value]
            else :
                probability = 0
                break
        final_probabilities[class_label] = probability

    predicted_class = max(final_probabilities,key = final_probabilities.get)
    return predicted_class,final_probabilities
    

data = pd.read_csv("naive.csv")
X = data.drop('buys_computer',axis = 1)
y = data['buys_computer']

tuple_to_classify = {}
print("Enter feature to classify : ")
for feature in X.columns:
    unique_values = X[feature].unique()
    print(f"Options for {feature} : {','.join(map(str,unique_values))}")
    value = input("Enter values to classify : ")
    tuple_to_classify[feature]=value

X_test = pd.DataFrame(tuple_to_classify)

predicted_class , final_probabilities = naive_bayes_classifier(X,y,X_test)
print(final_probabilities)
print(predicted_class)