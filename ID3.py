import pandas as pd
import numpy as np
from math import log2

# --- Step 1: Read Dataset ---
df = pd.read_csv("decision_tree.csv")
target_column = df.columns[-1]
features = list(df.columns[:-1])

# --- Step 2: Entropy Function ---
def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -sum(p * log2(p) for p in probabilities if p > 0)

# --- Step 3: Information Gain ---
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for val, count in zip(values, counts):
        subset = data[data[feature] == val]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (count / len(data)) * subset_entropy
    gain = total_entropy - weighted_entropy
    return gain

# --- Step 4: Build Decision Tree ---
def build_tree(data, features, target):
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    if len(features) == 0:
        return data[target].mode()[0]
    gains = {feature: information_gain(data, feature, target) for feature in features}
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}
    for val in np.unique(data[best_feature]):
        subset = data[data[best_feature] == val]
        subtree = build_tree(
            subset.drop(columns=[best_feature]),
            [f for f in features if f != best_feature],
            target
        )
        tree[best_feature][val] = subtree
    return tree

# --- Step 5: Print Decision Tree ---
def print_tree(tree, indent=" "):
    if not isinstance(tree, dict):
        print(indent + "_", tree)
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}[{feature}={val}]")
            print_tree(subtree, indent + "  ")

# --- Step 6: Print Conditional Probabilities ---
def print_conditional_probabilities(data, target):
    print("\nConditional Probabilities (P(feature=value | class)):")
    classes = np.unique(data[target])
    for feature in data.columns:
        if feature == target:
            continue
        print(f"\nFeature: {feature}")
        for cls in classes:
            subset = data[data[target] == cls]
            total = len(subset)
            for val in np.unique(data[feature]):
                count = len(subset[subset[feature] == val])
                prob = count / total if total > 0 else 0
                print(f"P({feature}={val} | {target}={cls}) = {prob:.3f}")

# --- Step 7: Prediction Function ---
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    feature_value = sample.get(feature)
    if feature_value not in tree[feature]:
        return "Unknown"
    subtree = tree[feature][feature_value]
    return predict(subtree, sample)

# --- Step 8: Main Execution ---
overall_entropy = entropy(df[target_column])
print(f"Total Entropy of dataset (INFO(D)) : {overall_entropy:.4f}")

gains = {}
for feature in features:
    gain = information_gain(df, feature, target_column)
    gains[feature] = gain
    print(f"Information gain for {feature} : {gain:.4f}")

best_feature = max(gains, key=gains.get)
print(f"\nFeature with highest information gain : {best_feature}")

decision_tree = build_tree(df, features, target_column)
print("\nDECISION TREE : ")
print_tree(decision_tree)

# Print conditional probabilities
print_conditional_probabilities(df, target_column)

# --- Step 9: User Input for Prediction ---
print("\nEnter feature values to predict class:")
sample = {}
for feature in features:
    options = list(df[feature].unique())
    print(f"Options for {feature}: {options}")
    sample[feature] = input(f"Enter {feature}: ")

predicted_class = predict(decision_tree, sample)
print(f"\nPredicted Class: {predicted_class}")
