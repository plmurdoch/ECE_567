import pandas
import joblib
import numpy
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def Training_Testing_Model(file, output_name):
    Parse_CSV = pandas.read_csv(file)
    Parse_CSV = Parse_CSV.replace([-numpy.inf, numpy.inf], numpy.nan)
    Parse_CSV = Parse_CSV.dropna()
    X_val = Parse_CSV.iloc[:,[2,4,5]+list(range(7,84))].values
    Y_val = Parse_CSV.iloc[:,-1].values
    Machine = KNeighborsClassifier(n_neighbors = 3) #3 good approximation without over fitting most accurate for KNN with most reliable accuracy 98%-99% with inf and nan entities dropped
    scale = StandardScaler()
    x_training, x_testing, y_training, y_testing = train_test_split(X_val, Y_val, test_size = 0.4)
    x_train = scale.fit_transform(x_training)
    x_test = scale.fit_transform(x_testing)
    for _ in range(10):
        Machine.fit(x_train, y_training)
        x_training, x_testing, y_training, y_testing = train_test_split(X_val, Y_val, test_size = 0.4)
        x_train = scale.fit_transform(x_training)
        x_test = scale.fit_transform(x_testing)
    y_prediction = Machine.predict(x_test)
    print(f"accuracy: {accuracy_score(y_prediction,y_testing)}")
    joblib.dump(Machine, output_name)

def main():
    if len(sys.argv) == 4:
        if sys.argv[1] == "develop":
            file_name = sys.argv[2]
            output_name = sys.argv[3]
            Training_Testing_Model(file_name, output_name)
        else:
            print("ERROR! Format input as:")
            print("python3 IDS.py develop input_csv_file output_joblib_file")
            exit(1)
    elif len(sys.argv) == 3:
        if sys.argv[1] == "deploy":
            exit(1)
        else:
            print("ERROR! Format input as:")
            print("python3 IDS.py deploy joblib_file")
        
    

if __name__ == "__main__":
    main()