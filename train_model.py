import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File paths
DATASET_FILE = "data/gestures_dataset.csv"
MODEL_FILE = "models/gesture_model_final_1.pkl"


def check_dataset_quality(df: pd.DataFrame) -> bool:
    """Print basic dataset quality report."""
    print("\n" + "=" * 50)
    print("DATASET QUALITY REPORT")
    print("=" * 50)

    print("\nShape:", df.shape)
    print("Columns:", list(df.columns))

    if "label" not in df.columns:
        print("\nERROR: 'label' column not found.")
        return False

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    print("\nExact duplicate rows:", df.duplicated().sum())

    feature_cols = [col for col in df.columns if col != "label"]
    duplicate_features = df[feature_cols].duplicated().sum()
    print("Duplicate feature rows (ignoring label):", duplicate_features)

    zero_variance_cols = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            zero_variance_cols.append(col)

    print("\nZero-variance feature columns:")
    if zero_variance_cols:
        print(zero_variance_cols)
    else:
        print("None")

    print("\nLabel percentages:")
    percentages = (df["label"].value_counts(normalize=True) * 100).round(2)
    print(percentages)

    print("\n" + "=" * 50)
    return True


def main():
    """Train RandomForest model and save it."""
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file not found: {DATASET_FILE}")
        return

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

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

    ok = check_dataset_quality(df)
    if not ok:
        return

    X = df.drop("label", axis=1)
    y = df["label"]

    print("\nTotal samples:", len(df))
    print("Number of features:", X.shape[1])
    print("Gesture classes:", sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")
    print("Testing Accuracy :", round(test_acc * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X,
        y,
        cv=5
    )

    print("\nCross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 15 Important Features:")
    print(feature_importances.head(15))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved as: {MODEL_FILE}")


if __name__ == "__main__":
    main()