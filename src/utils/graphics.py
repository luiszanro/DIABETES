import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotheatmap(data):
    df = data
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    
    # Plot the heatmap with improved readability
    plt.figure(figsize=(16, 12))  # Larger figure size for better readability
    sns.heatmap(correlation_matrix, 
                annot=True,  # Show correlation values
                fmt=".2f",   # Format numbers to 2 decimal places
                cmap='coolwarm', center=0,
                square=True, 
                annot_kws={"size": 10},  # Adjust font size for numbers
                cbar_kws={'shrink': 0.8})  # Shrink the color bar for a cleaner look
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels for clarity
    plt
    return correlation_matrix



def trainer_importance_plot(trainer_importance_sorted):
    
    colors = cm.viridis(np.linspace(0, 1, len(trainer_importance_sorted.head(10))))
    # Formatting figsize
    plt.figure(figsize=(10, 6))
    trainer_importance_sorted.head(10).plot(kind='bar', color=colors)
    # Formatting the plot
    plt.title("Top 10 Features (Logistic Regression)")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    plt
    topten = "Yes"
    return topten


def graph_recall_precision(recall, precision, auprc):
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {auprc:.2f})')  # Fixed unterminated f-string error
    plt.show()
    rp_img = "yes"
    return rp_img

def graph_cm(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes (0)", "Diabetes/Risk (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.show()
    cm_img = "yes"
    return cm_img
