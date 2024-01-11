import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np

# Read the balanced data
data_path = './data/train_data/balanced_data.csv'
balanced_data = pd.read_csv(data_path)

# Split features and labels
X = balanced_data.drop('label', axis=1)
y = balanced_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
rf_classifier = RandomForestClassifier()

# Initialize lists to store accuracies during training
train_accuracies = []
test_accuracies = []

# Set the range of number of estimators
n_estimators_range = range(10, 151, 10)

# Train the model and record accuracies
for n_estimators in n_estimators_range:
    rf_classifier.set_params(n_estimators=n_estimators)
    rf_classifier.fit(X_train, y_train)
    
    # Predict on the training and testing sets
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    
    # Calculate accuracies on the training and testing sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Save the accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
# Plot the training process accuracies
plt.plot(n_estimators_range, train_accuracies, label='Train Accuracy')
plt.plot(n_estimators_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Training Process Accuracy')
plt.legend()
plt.savefig('./Results/Graphs/RandomForest_results.png')
plt.show()

# Print the training metrics
best_index = np.argmax(test_accuracies)
best_n_estimators = n_estimators_range[best_index]
best_test_accuracy = test_accuracies[best_index]
print("Best Number of Estimators:", best_n_estimators)
print("Best Test Accuracy:", best_test_accuracy)

fpr_total =  cm[1][0] / (cm[1][0] + cm[1][1])
print(fpr_total)

mcc = matthews_corrcoef(y_test, y_pred)


print("Total FPR:", fpr_total)
print("Total MCC:", mcc)