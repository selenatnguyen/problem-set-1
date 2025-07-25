'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

df_test = pd.read_csv('data/df_arrests_test.csv')

calibration_plot(df_test['y'], df_test['pred_lr'], n_bins=5)
calibration_plot(df_test['y'], df_test['pred_dt'], n_bins=5)

print("Which model is more calibrated?")
print("Answer: logistic regression")

top50_lr = df_test.sort_values('pred_lr', ascending=False).head(50)
top50_dt = df_test.sort_values('pred_dt', ascending=False).head(50)

ppv_lr = top50_lr['y'].mean()
ppv_dt = top50_dt['y'].mean()

auc_lr = roc_auc_score(df_test['y'], df_test['pred_lr'])
auc_dt = roc_auc_score(df_test['y'], df_test['pred_dt'])

print("PPV for logistic regression:", ppv_lr)
print("PPV for decision tree:", ppv_dt)
print("AUC for logistic regression:", auc_lr)
print("AUC for decision tree:", auc_dt)
print("Do both metrics agree that one model is more accurate than the other?")
print("Answer: yes, logistic regression is more accurate")

