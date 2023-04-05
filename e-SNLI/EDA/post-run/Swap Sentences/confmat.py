import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn import metrics

font = {'size'   : 14}
matplotlib.rc('font', **font)

flip = pd.read_csv("./flip.csv")
nonflip = pd.read_csv("./nonflip.csv")

actual_f = flip['gold_label'].tolist()
predicted_f = flip['pred_label'].tolist()
actual_nf = nonflip['gold_label'].tolist()
predicted_nf = nonflip['pred_label'].tolist()

confusion_matrix_f = metrics.confusion_matrix(actual_f, predicted_f)
confusion_matrix_nf = metrics.confusion_matrix(actual_nf, predicted_nf)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_f, display_labels = ['contradiction', 'entailment', 'neutral'])
cm_display.plot()
plt.xlabel("Predicted label", weight="bold")
plt.ylabel("True label", weight="bold")
plt.show() 

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_nf, display_labels = ['contradiction', 'entailment', 'neutral'])
# cm_display.plot()
# plt.xlabel("Predicted label", weight="bold")
# plt.ylabel("True label", weight="bold")
# plt.show() 