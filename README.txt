ECE 567: Coding Submission
By: Payton Murdoch, V00904677 & Yun Ma, V01018599

---------------------------------------------------
How to run:
Our code consists of 2 scripts Test.py and Train.py
User running the code needs access to libraries:
-scikit-learn
-joblib
-numpy
-matplotlib
-pandas
-sys
---------------------------------------------------

Train.py
---------------------------------------------------
Command: python3 Train.py *input csv file*

Example: python3 Train.py flows_benign_and_DoS.csv

This code is utilized to create the ML model with
the Test data it outputs ML_model.joblib the tuned
ML.
---------------------------------------------------

Test.py
---------------------------------------------------
Command: python3 Test.py *joblib ML* *CSV file*

Example: python3 Test.py ML_model.joblib test.csv

This code is utilized to examine the test.csv file
Determine if it has true values or not, if not it
is our test which runs DoS attempts over HTTP and
normal SSH traffic to help with distinguishing true
values. After determining this the ML predicts Y
and proceeds to calculate FPR, Dr metrics and 
outputs them and the ROC curve with AUC using 
matplotlib and scikit functions.
---------------------------------------------------
                    **END**
