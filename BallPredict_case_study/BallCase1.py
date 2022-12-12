from sklearn import tree

Features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]] 

Labels = ["Tennis","Tennis","Criket","Tennis","Criket","Tennis","Criket","Tennis","Tennis","Tennis","Criket","Tennis","Criket","Tennis","Criket"]

obj = tree.DecisionTreeClassifier()

obj = obj.fit(Features, Labels)

print(obj.predict([[97,"Smooth"]]))