import pandas
import joblib
import numpy
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#CICFlowOrder, after parsing the columns from a pcap file converted from cicflowmeter we discerned that this list of columns would match the one that we constructed from the DoS dataset.
CICFlowOrder =[2,3,4,6,11,12,13,14,15,16,17,18,19,20,21,22,7,8,32,35,33,34,36,39,40,37,38,41,44,45,42,43,46,47,48,49,28,29,9,10,24,23,25,26,27,50,51,52,53,54,55,77,56,57,58,75,76,69,70,73,71,72,74,78,79,80,81,59,60,31,30,63,64,61,62,67,68,65,66]
#Training_Testing_Model takes the dataset file and output name of a joblib file as the inputs. This code is responsible for training the initial model Which abides by the following:
#First, for the input data we will discard any values that are negative or positive infinity and non-accessible numbers, to do this we convert all to numpy.nan (non-accessible number) and remove with the pandas .dropna() function.
#For actual data we remove the string entries in the table as described in the project parameters and one more column is removed as there is a repeat of FWD Header Length at index 61.
#For our ML model, we ran the ML model with numerous different classifiers and landed on utilizing KNN as it yielded the highest accuracy, we found that measuring the data point against 7 closest neighbours yielded the best results without over saturating the data.
def Training_Testing_Model(file, output_name):
    Parse_CSV = pandas.read_csv(file) #Read File
    Parse_CSV = Parse_CSV.replace([-numpy.inf, numpy.inf], numpy.nan) #swap any infinity with NaN.
    Parse_CSV = Parse_CSV.dropna() #Drop all NaN.
    Parse_CSV = Parse_CSV.replace(["BENIGN"], 0) #Binary rep of Benign as 0
    Parse_CSV = Parse_CSV.replace(["DoS Hulk","DoS Slowhttptest","DoS GoldenEye","DoS slowloris"], 1) #Binary rep of DoS as 1
    X_val = Parse_CSV.iloc[:,[2,4,5]+list(range(7,61))+list(range(62,84))].values #Collects DoS data for X
    Y_val = Parse_CSV.iloc[:,-1].values #Collects DoS label data for Y
    Machine = KNeighborsClassifier(n_neighbors = 7) #7 good approximation without over fitting most accurate for KNN with most reliable accuracy 95-97% with inf and nan entities dropped
    scale = StandardScaler() #Standard scaler for optimizing inputs
    x_train = scale.fit_transform(X_val) #standardize training
    Machine.fit(x_train, Y_val) #Train the machine
    joblib.dump(Machine, output_name) #Output ML model as joblib file.

def True_Values(CSV):
    true_values = []
    src_prt = CSV.iloc[:,2].values
    dest_prt = CSV.iloc[:,3].values
    for x in range(len(CSV)):
        if src_prt[x] != 80 and dest_prt[x] != 80:
            true_values.append(0)
        else:
            true_values.append(1)
    CSV['True Label'] = true_values
    CSV.to_csv('Output_flow.csv',index=False) #output as Output_flow.csv file using pandas

#Validating_Data takes the CSV validating dataset and the joblib ML file name, it loads the ML into the variable machine, then it loads the csv file into two dataframes. 
#The first dataframe remains the same and is used to output a modified csv with a label column for visual representation of columns reported as BENIGN or DoS.
#The second Dataframe is the validation data manipulated so that the columns match on the same order as the training sets and transformed to fit the same standardized input.
def Validating_Data(file, input):
    Machine = joblib.load(input) #Load ML joblib dump as machine
    initial_CSV = pandas.read_csv(file) #store initial dataset
    Parse_CSV = pandas.read_csv(file) #dataset to be manipulated
    X_val = Parse_CSV.iloc[:,CICFlowOrder].values #grab values in the order to match normal input variables
    scale = StandardScaler() #Load standard scalar
    x_test = scale.fit_transform(X_val) #scaling x_testing data
    y_prediction = Machine.predict(x_test) #validation data
    column = [] #initializing empty column
    for i in y_prediction: #for loop iterating over prediction
        column.append(i) #Or BENIGN
    initial_CSV['Label'] = column #Adds new column to CSV datagframe
    True_Values(initial_CSV)


def main(): #Main function where functionality is determined
    if len(sys.argv) == 4: #Must be length 4
        if sys.argv[1] == "develop": #Develop, for training ML and saving
            file_name = sys.argv[2] #csv_file is 2nd argument
            output_name = sys.argv[3] #joblib file name is 3rd argument
            Training_Testing_Model(file_name, output_name) #Pass to function
        elif sys.argv[1] == "deploy": #Deploy, for loading and running ML on validation set
            file_name = sys.argv[2] #CSV file is 2nd argument
            input_name = sys.argv[3] #joblib filename is 3rd argument (Must exist (no error checking implemented))
            Validating_Data(file_name, input_name) #Pass to validation function
    else: #Showing proper input options.
        print("ERROR! Format input as:")
        print("python3 IDS.py develop input_csv_file output_joblib_file")
        print("//This is for initial Model Training and Testing//")
        print("OR")
        print("python3 IDS.py deploy input_csv_file input_joblib_file")
        print("//This is for model deployment and validation//")
        exit(1)
            
if __name__ == "__main__":
    main()