# Brief Description:  
This directory contains the following files:  
- `Assignment2_Gr_A.pdf`: Description of problem statement  
- `wine_data.csv`: Contains the data used for clustering and training the SVM Classifier and MLP Classifier.
- `q1.py`: Contains the solution to the Question 1: Unsupervised Learning
- `q2.py`: Contains the solution to the Question 2: Supervised Learning
- `requirements_1.txt`: Contains all the necessary dependencies and their versions for Question 1
- `requirements_2.txt`: Contains all the necessary dependencies and their versions for Question 2
- `simulation1.txt`: The Output of the First Question.(`q1.py`)
- `simulation2.txt`: The Output of the Second Question.(`q2.py`)

# Directions to use the code  
1. Download this directory into your local machine

2. Copy the wine_data.csv file in the Source Code Directory if absent, change the name of the file by changing the filename variable at the top of each Python File.

3. Ensure all the necessary dependencies with required version and latest version of Python3 are available (verify with `requirements_{q}.txt`)  <br>
 `pip3 install -r requirements_{q}.txt` where q = 1,2.
4. The relevant output will be displayed on the terminal or console and saved as simulation1.txt for Question 1 and simulation2.txt for Question 2 respectively.

# Plots and Other Results
1. For Question 1 there are three plots generated:-
   1. `variance_ratio_pca.png` :- It's the Plot of the Fraction of Variance Explained for Each Component by Principal Component Analysis 
   2. `variance_ratio_cumulative sum.png` :- It's the Plot the Cumulative Fraction Sum of Variance Explained for Each Component by Principal Component Analysis. It also marks the component where 95% of Variance is explained.
   3. `k_vs_nmi.png`:- It plots the value of K-the number of clusters vs Normalised Mutual Information Score.
   4. `simulation1.txt`: The Output of the First Question(`q1.py`) is copied into this file.
2. For Question 2 there is one plot generated:- 
   1. `learning_rate_vs_acc.png`:- It plots the learning rate vs. accuracy for the MLP Classifier. 
   2. `simulation2.txt`: The Output of the Second Question(`q2.py`) is copied into this file.



