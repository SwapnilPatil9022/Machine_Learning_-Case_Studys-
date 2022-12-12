# import requried libarary
from sklearn import tree

# Load the dataset
# Rough		1
# Smooth	0

# Tennis	1
# Criket 	2

def BallPredictor(weight,surface):
	Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]] 
	Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

	#Deside the Machine learning algorithem
	obj = tree.DecisionTreeClassifier()

	# Perform the treanning of model
	obj = obj.fit(Features, Labels)

	# perform of testing 
	ret = (obj.predict([[weight,surface]]))
	if ret == 1:
		print("Your objects looks like Tennis ball")
	else:
		print("Your objects look like Criket ball")

def main():
	print("-------------Ball Predictor Case Study------------")

	print("Please enter the weight your objects in grams")
	weight = int(input())

	print("Please enter the type of surface of your objects (Rough / Smooth)")
	surface = input()

	if surface.lower() == "rough":
		surface = 1
	elif surface.lower() == "smooth":
		surface = 0
	else:
		print("Invalid type of surface")
		exit()

	BallPredictor(weight,surface)

if __name__ == "__main__":
	main()