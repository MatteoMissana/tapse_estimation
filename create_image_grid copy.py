import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = np.array([[7, 1],
               [2, 0]])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[">35 % (Normal)", "<35 % (Reduced)"])
disp.plot(cmap='Blues', values_format='d', colorbar=False)

plt.title("Confusion Matrix for RVFAC - based RV evaluation")

# Save as PDF
plt.tight_layout()
plt.savefig(r"D:\mmissana\tapse_estimation\2d\results\confusion_matrices\confusion_matrix_rvfac.pdf", format='pdf', bbox_inches='tight')

# Optional: also show it interactively
plt.show()
