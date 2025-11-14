import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load extracted features and labels
X = np.load('features.npy')          # Shape: (2000, 30, 1280)
y = np.load('features_labels.npy')   # Shape: (2000,)

print("✅ Data loaded:", X.shape, y.shape)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN + GRU model
model = Sequential([
    GRU(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
print("🚀 Training started...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.2f}")

# Classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the model
model.save('crime_detection_model_gru.keras')
print("✅ Model saved as 'crime_detection_model_gru.keras'")
