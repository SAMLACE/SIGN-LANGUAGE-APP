# Import the pickle module for loading and saving serialized objects.
import pickle

# Import the RandomForestClassifier from scikit-learn's ensemble module for classification tasks(improve predictive performance by combining the strengths of multiple models.).
from sklearn.ensemble import RandomForestClassifier

# Import the train_test_split function from scikit-learn to split the data into training and testing sets(tuning hyperparameters, and understanding model performance).
from sklearn.model_selection import train_test_split

# Import the accuracy_score function from scikit-learn to evaluate the model's accuracy.
from sklearn.metrics import accuracy_score

# Import numpy as np for numerical operations and array handling.
import numpy as np


# Load the preprocessed data and labels from the 'data.pickle' file.
data_dict = pickle.load(open('./data.pickle', 'rb')) 

# Convert the loaded data and labels from lists to numpy arrays for efficient computation.
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets.
# - x_train, y_train: Training data and labels.
# - x_test, y_test: Testing data and labels.
# - test_size=0.2: 20% of the data will be used for testing, 80% for training.
# - shuffle=True: The data will be shuffled before splitting.
# - stratify=labels: The split will maintain the distribution of classes.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier model.
model = RandomForestClassifier()

# Train the RandomForestClassifier model using the training data and labels.
model.fit(x_train, y_train)

# Use the trained model to make predictions on the testing data.
y_predict = model.predict(x_test)

# Calculate the accuracy of the model by comparing the predicted labels with the actual labels.
score = accuracy_score(y_predict, y_test)

# Print the accuracy of the model as a percentage.
print('{}% of samples were classified correctly !'.format(score * 100))

# Open a file named 'model.p' in binary write mode to save the trained model.
f = open('model.p', 'wb')

# Use pickle to serialize the trained model and save it to the file.
pickle.dump({'model': model}, f)

# Close the file after writing to it.
f.close()
