import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Encode categorical variables
label_encoders = {}
for column in ['Crop', 'Soil_Type', 'Weather']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data.drop('Best_Fertilizer', axis=1)
y = data['Best_Fertilizer']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(label_encoders['Best_Fertilizer'].classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
# Evaluate the model on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Example new data point
new_data = pd.DataFrame({
    'Crop': [label_encoders['Crop'].transform(['Corn'])[0]],
    'Soil_Type': [label_encoders['Soil_Type'].transform(['Clay'])[0]],
    'Weather': [label_encoders['Weather'].transform(['Rainy'])[0]]
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Predict the best fertilizer
best_fertilizer_index = np.argmax(model.predict(new_data_scaled), axis=-1)
best_fertilizer = label_encoders['Best_Fertilizer'].inverse_transform(best_fertilizer_index)[0]

print(f"The best fertilizer for the given conditions is: {best_fertilizer}")
