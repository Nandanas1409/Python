import pandas as pd
import numpy as np
from math import log2

df = pd.read_csv('decision_tree.csv')
target_column = df.columns[-1]
features = df.columns[:-1]

def entropy(labels) :
    values,counts = np.unique(labels,return_counts=True)
    probabilities = counts/counts.sum()
    return -sum(p*log2(p) for p in probabilities if p>0)

def gain_ratio(data,feature,target) :
    total_entropy = entropy(data[target])
    values,counts = np.unique(data[feature],return_counts=True)
    weighted_entropy = 0
    split_info =0
    for val,count in zip(values,counts) :
        subset = data[data[feature]==val]
        subset_entropy = entropy(subset[target])
        weight = count/len(data)
        weighted_entropy += weight * subset_entropy
        split_info -= weight*log2(weight) if weight>0 else 0
    info_gain = total_entropy - weighted_entropy
    if split_info==0:
        return 0
    return info_gain/split_info

def build_tree(data,features,target) :
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    if len(features) == 0:
        return data[target].mode()[0]
    gain = {feature : gain_ratio(data,feature,target) for feature in features}
    best_feature = max(gain,key=gain.get)
    tree = {best_feature:{}}
    for val in np.unique(data[best_feature]):
        subset = data[data[best_feature]==val]
        subtree = build_tree(
            subset.drop(columns=[best_feature]),
            [f for f in features if f!= best_feature],
            target
        )
        tree[best_feature][val]=subtree
    return tree

def print_tree(tree,indent=" ") :
    if not isinstance(tree,dict) :
        print(indent + "_",tree)
        return
    for feature,branches in tree.items():
        for val,subtree in branches.items():
            print(f"{indent}[{feature} = {val}]")
            print_tree(subtree,indent+" ")

def conditional_prob(data,target) :
    print("Conditional Probabilities : ")
    classes = data[target].unique()
    for feature in data.columns:
        if feature==target:
            continue
        print(f"Feature : {feature}")
        for cls in classes:
            subset = data[data[target]==cls]
            total = len(subset)
            for val in np.unique(data[feature]) :
                count = len(subset[subset[feature]==val])
                prob = count/total if total>0 else 0
                print(f"P({feature}={val}|{target}={cls}) : {prob:.4f}")

overall_entropy = entropy(df[target_column])
print(f"Total entropy : {overall_entropy}")

gains = {}
for feature in features:
    gain = gain_ratio(df,feature,target_column)
    gains[feature]=gain
    print(f"'Gain ratio of {feature} : {gain}")

best_feature = max(gains,key=gains.get)
print("Feature with higehst gain ratio : ",best_feature)

tree = build_tree(df,features,target_column)
print_tree(tree)

conditional_prob(df,target_column)
