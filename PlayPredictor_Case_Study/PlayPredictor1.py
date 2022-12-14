from sklearn import tree

def PlayPredictorCasestudy(whether,temprature):
    # 3 entries
    # Sunny    - 1
    # Overcast - 2
    # Rainy    - 3

    # temprature are
    # Cold   - 5
    # Mild   - 6
    # Hot    - 7

    # Labels 
    # Yes - 1
    # No -  0

    # 1 Load the data
    Features = [[1,7],[1,7],[2,7],[3,6],[3,5],[3,5],[2,5],[1,6],[1,5],[3,6],[1,6],[2,6],[2,7],[3,6],[3,6],[3,5],[3,5],[2,5],[1,6],[1,5],[3,6],[1,6],[1,7],[1,7],[2,7],[3,6],[3,5],[2,5],[1,6],[1,5]]    

    Labels = [0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,1,1,1,1,0,1]

    #Deside the Machine learning algorithem
    obj = tree.DecisionTreeClassifier()

    # Perform the treanning of model
    obj = obj.fit(Features, Labels)

    # perform of testing 
    ret = (obj.predict([[whether,temprature]]))
    if ret == 1:
        print("Now you are play.")
    else:
        print("You not play.")

def main():
    print("-------------Play Predictor Case Study------------")

    print("Please enter the whether (Sunny / Overcast / Rainy)")
    whether = input()

    if whether.lower() == "sunny":
        whether = 1
    elif whether.lower() == "overcast":
        whether = 2
    elif whether.lower() == "rainy":
        whether = 3
    else:
        print("Invalid type of whether")
        exit()

    print("Please enter the temprature (Cold / Mild / Hot)")
    temprature = input()

    if temprature.lower() == "cold":
        temprature = 5
    elif temprature.lower() == "mild":
        temprature = 6
    elif temprature.lower() == "hot":
        temprature = 7
    else:
        print("Invalid type of temprature")
        exit()

    PlayPredictorCasestudy(whether,temprature)

if __name__ == "__main__":
    main()