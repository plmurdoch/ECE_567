import pandas
import joblib
import numpy
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def Training_Testing_Model(file, output_name):
    Parse_CSV = pandas.read_csv(file)
    Parse_CSV = Parse_CSV.replace([-numpy.inf, numpy.inf], numpy.nan)
    Parse_CSV = Parse_CSV.dropna()
    Parse_CSV = Parse_CSV.replace(["BENIGN"], 0)
    Parse_CSV = Parse_CSV.replace(["DoS Hulk","DoS Slowhttptest","DoS GoldenEye","DoS slowloris"], 1)
    X_val = Parse_CSV.iloc[:,list(range(7,84))].values
    Y_val = Parse_CSV.iloc[:,-1].values
    Machine = KNeighborsClassifier(n_neighbors = 7) #7 good approximation without over fitting most accurate for KNN with most reliable accuracy 95-97% with inf and nan entities dropped
    scale = StandardScaler()
    x_training, x_testing, y_training, y_testing = train_test_split(X_val, Y_val, test_size = 0.2, stratify=Y_val)
    x_train = scale.fit_transform(x_training)
    x_test = scale.fit_transform(x_testing)
    Machine.fit(x_train, y_training)
    y_pred = Machine.predict(x_test)
    print(f"Model Accuracy:{accuracy_score(y_testing, y_pred)}")
    RocCurveDisplay.from_predictions(y_testing, y_pred)
    plt.show()
    joblib.dump(Machine, output_name)

def Validating_Data(file, input):
    Machine = joblib.load(input)
    Parse_CSV = pandas.read_csv(file)

def main():
    if len(sys.argv) == 4:
        if sys.argv[1] == "develop":
            file_name = sys.argv[2]
            output_name = sys.argv[3]
            Training_Testing_Model(file_name, output_name)
        elif sys.argv[1] == "deploy":
            file_name = sys.argv[2]
            input_name = sys.argv[3]
            Validating_Data(file_name, input_name)
    else:
        print("ERROR! Format input as:")
        print("python3 IDS.py develop input_csv_file output_joblib_file")
        print("//This is for initial Model Training and Testing//")
        print("OR")
        print("python3 IDS.py deploy input_csv_file input_joblib_file")
        print("//This is for model deployment and validation//")
        exit(1)
            
        
    

if __name__ == "__main__":
    main()