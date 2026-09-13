import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_FILE = "data/gestures_dataset.csv"
MODEL_FILE = "models/gesture_model_svm.pkl"


def main():
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file not found: {DATASET_FILE}")
        return

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(DATASET_FILE)

    print("\nDataset preview:")
    print(df.head())

    print("\nColumns:")
    print(df.columns)

    if df.empty:
        print("Dataset is empty.")
        return

    if "label" not in df.columns:
        print("Dataset must contain a 'label' column.")
        return

    # -----------------------------
    # Features & labels
    # -----------------------------
    X = df.drop("label", axis=1)
    y = df["label"]

    print("\nTotal samples:", len(df))
    print("Number of features:", X.shape[1])
    print("Gesture classes:", sorted(y.unique()))

    print("\nSamples per class:")
    print(y.value_counts())

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -----------------------------
    # SVM Model
    # -----------------------------
    model = SVC(
        kernel="rbf",       # best for non-linear data
        C=10,               # regularization
        gamma="scale",      # automatic scaling
        probability=True,   # required for confidence scores
        random_state=42
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Accuracy
    # -----------------------------
    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")
    print("Testing Accuracy :", round(test_acc * 100, 2), "%")

    # -----------------------------
    # Reports
    # -----------------------------
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------------
    # Cross-validation
    # -----------------------------
    cv_scores = cross_val_score(
        SVC(kernel="rbf", C=10, gamma="scale"),
        X,
        y,
        cv=5
    )

    print("\nCross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

    # -----------------------------
    # Save model
    # -----------------------------
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved as: {MODEL_FILE}")


if __name__ == "__main__":
    main()