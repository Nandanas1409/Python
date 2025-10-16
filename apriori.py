import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
with open("/Users/nandanasnair/Desktop/Nandana/Python/file.csv","r") as f:
    transactions = [line.strip().split(",") for line in f.readlines()]
    print(transactions)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array,columns=te.columns_)
print(df)
min_support = float(input("Enter minimum support : "))
frequent_itemsets = apriori(df,min_support=min_support,use_colnames=True)

print("Frequent Itemsets : \n")
print(frequent_itemsets)

min_confidence = float(input("enter minimum confiidence : "))
rules = association_rules(frequent_itemsets,metric = 'confidence',min_threshold = min_confidence)
rules = rules[['antecedents','consequents','support','confidence']]
print(rules)
