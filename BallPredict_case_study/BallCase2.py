from sklearn import tree

# Load the dataset
# Rough		1
# Smooth	0

# Tennis	1
# Criket 	2

Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]] 
Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

#Deside the Machine learning algorithem
obj = tree.DecisionTreeClassifier()

# 
obj = obj.fit(Features, Labels)

print(obj.predict([[97,0],[35,1]]))