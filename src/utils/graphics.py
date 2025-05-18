import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotheatmap(data, output_dir):
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
    output_dir = record_on_output(output_dir)
    filename = "Heatmap_plot.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    #plt
    plt.close()
    print("Plotheatmap is completed ")
    return correlation_matrix, file_path



def trainer_importance_plot(trainer_importance_sorted,output_dir):
    
    colors = cm.viridis(np.linspace(0, 1, len(trainer_importance_sorted.head(10))))
    # Formatting figsize
    plt.figure(figsize=(10, 6))
    trainer_importance_sorted.head(10).plot(kind='bar', color=colors)
    # Formatting the plot
    plt.title("Top 10 Features (Logistic Regression)")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    output_dir = record_on_output(output_dir)
    filename = "trainer_importance_plot.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    #plt
    plt.close()
    print("Plotheatmap is completed ")
    topten = "Yes"
    return topten, file_path


def graph_recall_precision(recall, precision, auprc, output_dir):
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {auprc:.2f})')  # Fixed unterminated f-string error
    #plt.show()
    output_dir = record_on_output(output_dir)
    filename = "Graph_recall_precision_voting_1.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    #plt
    plt.close()
    print("Graph_recall_precision_voting_1 is completed ")
    rp_img = "yes"
    return rp_img, file_path

def graph_cm(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes (0)", "Diabetes/Risk (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.show()
    cm_img = "yes"
    return cm_img

def record_on_output(output_dir):
    if output_dir is "ouput/":
        output_dir = "../../output/"  # Default path
    ##output_dir = "../../output/" 
    return output_dir