import pandas
import joblib
import numpy
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler

#Training_Model takes the dataset file as the input. This code is responsible for training the initial model Which abides by the following:
#First, for the input data we will discard any values that are negative or positive infinity and non-accessible numbers, to do this we convert all to numpy.nan (non-accessible number) and remove with the pandas .dropna() function.
#For actual data we remove the string entries in the table as described in the project parameters and one more column is removed as there is a repeat of FWD Header Length at index 61.
#We also decided to switch our data over to a binary representation where benign is 0 and any DoS is 1.
#For our ML model, we ran the ML model with numerous different classifiers and landed on utilizing RandomForestClassifier as it yielded the highest accuracy in a training/testing split.
#While computing relevant information RandomForestClassifier also had a built in feature_importance_ function which allowed us to eliminiate data columns which served no importances.
def Training_Model(file):
    Parse_CSV = pandas.read_csv(file) #Read File
    Parse_CSV = Parse_CSV.replace([-numpy.inf, numpy.inf], numpy.nan) #swap any infinity with NaN.
    Parse_CSV = Parse_CSV.dropna() #Drop all NaN.
    Parse_CSV = Parse_CSV.replace(["BENIGN"], 0) #Binary rep of Benign as 0
    Parse_CSV = Parse_CSV.replace(["DoS Hulk","DoS Slowhttptest","DoS GoldenEye","DoS slowloris"], 1) #Binary rep of DoS as 1
    X_val = Parse_CSV.iloc[:,[2,4,5]+list(range(7,37))+list(range(40,50))+[52,53,54]+list(range(57,61))+list(range(68,84))].values #Collects DoS data for X
    Y_val = Parse_CSV.iloc[:,-1].values #Collects DoS label data for Y
    x_training,x_testing, y_training, y_testing = train_test_split(X_val, Y_val,test_size=0.2, stratify=Y_val) #Split our X and Y into an 80/20 train test split.
    Machine = RandomForestClassifier(n_estimators=50, random_state = 0) #RandomForest with 50 estimators which were experimentally selected.
    scale = MaxAbsScaler() #MaxAbs scaler for optimizing inputs through experimental selection.
    x_train = scale.fit_transform(x_training) #standardize training
    x_test = scale.fit_transform(x_testing) #standardize the testing set.
    Machine.fit(x_train, y_training) #Train the machine
    y_experiment = Machine.predict(x_test) #Predict y_values
    print(f"Accuracy score: {accuracy_score(y_testing, y_experiment)}") #Measure and print out general accuracy score of machine.
    joblib.dump(Machine, "ML_model.joblib") #Output ML model as joblib file.


def main(): #Main function where functionality is determined
    if len(sys.argv) == 2: #Must be length 2
            file_name = sys.argv[1] #csv_file is only argument
            Training_Model(file_name) #Pass to function
    else: #Showing proper input options.
        print("ERROR! Format input as:")
        print("python3 Train.py input_csv_file")
        print("//This is for initial Model Training//")
        exit(1)

if __name__ == "__main__":
    main()