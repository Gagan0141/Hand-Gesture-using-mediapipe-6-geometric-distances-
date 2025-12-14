import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

FEATURES_PATH = "features/features.csv"
MODEL_OUT = "models/tf_gesture_model.keras"
LABELS_OUT = "models/labels.txt"

df = pd.read_csv(FEATURES_PATH)
X = df.drop("label", axis=1).values
y = df["label"].values

encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation="relu", input_dim=X.shape[1]),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(y.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)

with open(LABELS_OUT, "w") as f:
    f.write("\n".join(encoder.classes_))
