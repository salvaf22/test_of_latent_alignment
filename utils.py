import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

def calculate_advanced_metrics(y_true, y_pred, class_names=None):
    """
    Calculates and prints a comprehensive report of advanced clinical metrics.
    Supports both binary and multi-class classification.
    """
    print("\n" + "="*55)
    print("ADVANCED CLINICAL METRICS REPORT")
    print("="*55)

    cm = confusion_matrix(y_true, y_pred)

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nCOHEN'S KAPPA: {kappa:.4f} ", end="")
    if kappa > 0.8: print("(Excellent Agreement)")
    elif kappa > 0.6: print("(Good Agreement)")
    elif kappa > 0.4: print("(Moderate Agreement)")
    elif kappa > 0.2: print("(Fair Agreement)")
    else: print("(Slight/Poor Agreement)")

    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nSENSITIVITY (Recall Class 1): {sensitivity*100:.2f}%")
        print(f"   (Ability to correctly identify target events)")
        print(f"\nSPECIFICITY (Recall Class 0): {specificity*100:.2f}%")
        print(f"   (Ability to ignore background noise / avoid false alarms)")
    else:
        specificities = []
        for i in range(len(np.unique(y_true))):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm[:, i], i))
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        print(f"\nMACRO-AVERAGED SPECIFICITY: {np.mean(specificities)*100:.2f}%")

    print("\nDETAILED REPORT (Scikit-Learn):")
    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))
    print("="*55)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Generates a colored Heatmap ready for publication.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 12})
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.ylabel('Actual Class', fontsize=12, labelpad=10)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()