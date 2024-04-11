import pandas
import joblib
import numpy
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#CICFlowOrder, after parsing the columns from a pcap file converted from cicflowmeter we discerned that this list of columns would match the one that we constructed from the DoS dataset.
#Utilizing the unofficial python wrapper for CICFlowmeter, as we could not get the official product to work on kali.
CICFlowOrder =[2,3,4,6,11,12,13,14,15,16,17,18,19,20,21,22,7,8,32,35,33,34,36,39,40,37,38,41,44,45,42,43,46,47,48,49,28,29,9,10,24,23,25,26,27,50,51,52,53,54,55,77,56,57,58,75,76,69,70,73,71,72,74,78,79,80,81,59,60,31,30,63,64,61,62,67,68,65,66]

#For our testing parameters, we ran the dos attack while conducting SSH traffic on the same server, thus we can isolate the true labels through the use of determining http ports were in use by the attacker.
def True_Values(CSV):
    true_values = []
    src_prt = CSV.iloc[:,2].values
    dest_prt = CSV.iloc[:,3].values
    for x in range(len(CSV)):
        if dest_prt[x] != 80:
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
    X_val = initial_CSV.iloc[:,CICFlowOrder].values #grab values in the order to match normal input variables
    scale = StandardScaler() #Load standard scalar
    x_test = scale.fit_transform(X_val) #scaling x_testing data
    y_prediction = Machine.predict(x_test) #validation data
    true_values = []
    if "Label" not in initial_CSV: #If we need to find the true values
        column = [] #initializing empty column
        for i in y_prediction: #for loop iterating over prediction
            column.append(i) #Or BENIGN
        initial_CSV['Label'] = column #Adds new column to CSV datagframe
        initial_CSV.to_csv('Output_flow.csv',index=False) #output as Output_flow.csv file using pandas
        true_values = True_Values(initial_CSV) #If this file is straight from CICFlowMeter then we will get true labels with True_values function.
    else:
        initial_CSV = initial_CSV.replace(["BENIGN"], 0) #Binary rep of Benign as 0
        initial_CSV = initial_CSV.replace(["DoS Hulk","DoS Slowhttptest","DoS GoldenEye","DoS slowloris"], 1) #Binary rep of DoS as 1
        test_values = initial_CSV[:,-1].values
        for i in test_values:
            true_values.append(i)
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for i in range(len(y_prediction)):
        if y_prediction[i] == true_values[i]:
            if true_values[i] == 0:
                TN = TN+1
            else:
                TP = TP+1
        else:
            if true_values[i] == 1:
                FN = FN+1
            else:
                FP = FP+1
    FPR = (FP/(FP+TN))
    DR = (TP/(TP+FN))
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