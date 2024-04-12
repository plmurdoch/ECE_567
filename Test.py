import pandas
import joblib
import sys
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


#For our testing parameters, we ran the dos attack while conducting SSH traffic on the same server, thus we can isolate the true labels through the use of determining http ports were in use as the dest port from the attacker.
def True_Values(CSV):
    true_values = [] #True value array
    dest_prt = CSV.iloc[:,4].values #Get destination port values.
    for x in range(len(dest_prt)): #Iterating through the total dest ports
        if dest_prt[x] != 80: #if the taget port is not HTTP 
            true_values.append(0)
        else:
            true_values.append(1)
    return true_values

#Validating_Data takes the CSV validating dataset and the joblib ML file name, it loads the ML into the variable machine, then it loads the csv file into two dataframes. 
#The first dataframe remains the same and is used to output a modified csv with a label column for visual representation of columns reported as BENIGN or DoS.
#The second Dataframe is the validation data manipulated so that the columns match on the same order as the training sets and transformed to fit the same standardized input.
def Validating_Data(file, input):
    Machine = joblib.load(input) #Load ML joblib dump as machine
    initial_CSV = pandas.read_csv(file) #store initial dataset
    X_val = initial_CSV.iloc[:,[2,4,5]+list(range(7,37))+list(range(40,50))+[52,53,54]+list(range(57,62))+list(range(68,83))].values #grab values in the order to match normal input variables
    scale = MaxAbsScaler() #Load standard scalar
    x_test = scale.fit_transform(X_val) #scaling x_testing data
    y_prediction = Machine.predict(x_test) #validation data
    true_values = []
    if initial_CSV.isin(["No Label"]).any().any(): #If we need to find the true values
        initial_CSV = initial_CSV.drop(columns = ["Label"])
        column = [] #initializing empty column
        for i in y_prediction: #for loop iterating over prediction
            column.append(i) #Or BENIGN
        initial_CSV['Label'] = column #Adds new column to CSV datagframe
        initial_CSV.to_csv('Output_flow.csv',index=False) #output as Output_flow.csv file using pandas
        true_values = True_Values(initial_CSV) #If this file is straight from CICFlowMeter then we will get true labels with True_values function.
    else:
        initial_CSV = initial_CSV.replace(["BENIGN"], 0) #Binary rep of Benign as 0
        initial_CSV = initial_CSV.replace(["DoS Hulk","DoS Slowhttptest","DoS GoldenEye","DoS slowloris"], 1) #Binary rep of DoS as 1
        test_values = initial_CSV.iloc[:,-1].values #Grabs last column of labelled csv file
        for i in test_values: #iterate through dataframe columns
            true_values.append(i) #Append to a simple list.
    FP = 0 #False Positives
    FN = 0 #False Negatives
    TP = 0 #True Positives
    TN = 0 #True Negatives
    for i in range(len(y_prediction)): #Iterating through the prediction array.
        if y_prediction[i] == true_values[i]: #Comparing predicted values with the true values
            if true_values[i] == 0: #If they match on Negative then True negative.
                TN = TN+1
            else: #If they match on Positive then true positive
                TP = TP+1
        else: #If they do not match
            if true_values[i] == 1: #If true value is an attack
                FN = FN+1 #False negative
            else: #If true value is not an attack 
                FP = FP+1 #Then False positive.
    FPR = (FP/(FP+TN)) #False Positive Rates
    DR = (TP/(TP+FN)) #False Negatvie Rates
    print("----------------------------------------------")
    print(f"False Positive Rate: {FPR*100}%")
    print(f"Detection Rate: {DR*100}%")
    print("----------------------------------------------")
    fpr,tpr,_=roc_curve(true_values,y_prediction)
    auc = roc_auc_score(true_values,y_prediction)
    plt.plot(fpr,tpr,label="AUC=%.2f"%auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def main(): #Main function where functionality is determined
    if len(sys.argv) == 3: #Must be length 3
        input_name = sys.argv[1] #joblib filename is 1st argument (Must exist (no error checking implemented))
        file_name = sys.argv[2] #CSV file is 2nd argument
        Validating_Data(file_name, input_name) #Pass to validation function
    else: #Showing proper input options.
        print("ERROR! Format input as:")
        print("python3 Test.py input_joblib_file input_csv_file")
        print("//This is for model deployment and validation//")
        exit(1)
            
if __name__ == "__main__":
    main()