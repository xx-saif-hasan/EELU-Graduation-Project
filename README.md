!pip install catboost --quiet

import re
import json
import math
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# -----------------------------------------
# 1️⃣ تحميل البيانات
# -----------------------------------------

file_path = "/kaggle/input/datasets/saifullahhasan/data-ready/pisa2022_columns_named.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# -----------------------------------------
# 2️⃣ حساب متوسط درجات الرياضيات
# -----------------------------------------

math_cols = [f"PV{i}MATH" for i in range(1, 11) if f"PV{i}MATH" in df.columns]
df["math_score"] = df[math_cols].mean(axis=1)

# -----------------------------------------
# 3️⃣ تحويلها إلى تصنيفات
# -----------------------------------------

def categorize(score):
    if pd.isna(score):
        return None
    elif score < 300:
        return "Fail"
    elif score < 400:
        return "Pass"
    elif score < 500:
        return "Good"
    else:
        return "Excellent"

df["Performance_Level"] = df["math_score"].apply(categorize)
df = df.dropna(subset=["Performance_Level"])

# -----------------------------------------
# 4️⃣ تجهيز Features و Target
# -----------------------------------------

drop_cols = math_cols + ["math_score"]
X = df.drop(columns=drop_cols + ["Performance_Level"], errors="ignore")
y = df["Performance_Level"]

print("Total features before selection:", X.shape[1])

# -----------------------------------------
# 5️⃣ تحويل الأعمدة الرقمية قليلة القيم إلى categorical بشكل مضبوط
# -----------------------------------------

CATEGORICAL_NUMERIC_THRESHOLD = 5

def convert_low_cardinality_numeric_to_categorical(df_in, threshold=5):
    df_out = df_in.copy()
    converted_cols = []

    for col in df_out.columns:
        s = df_out[col]

        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()

            if len(non_null) == 0:
                continue

            unique_count = non_null.nunique()

            # يتحول فقط لو عدد القيم قليل والقيم integer-like
            if unique_count <= threshold:
                is_integer_like = np.all(np.isclose(non_null, np.round(non_null)))

                if is_integer_like:
                    df_out[col] = s.apply(
                        lambda x: str(int(round(x))) if pd.notna(x) else "Missing"
                    )
                    converted_cols.append(col)

    return df_out, converted_cols

X, converted_to_cat_cols = convert_low_cardinality_numeric_to_categorical(
    X,
    threshold=CATEGORICAL_NUMERIC_THRESHOLD
)

print("Converted low-cardinality numeric columns to categorical:", len(converted_to_cat_cols))

# -----------------------------------------
# 6️⃣ تقسيم أولي لعمل feature selection
# -----------------------------------------

X_train_fs_raw, X_test_fs_raw, y_train_fs_text, y_test_fs_text = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# خلي كل categorical نص ومفيش NaN
for col in X_train_fs_raw.select_dtypes(include=["object"]).columns:
    X_train_fs_raw[col] = X_train_fs_raw[col].fillna("Missing").astype(str)
    X_test_fs_raw[col] = X_test_fs_raw[col].fillna("Missing").astype(str)

categorical_features_all = X_train_fs_raw.select_dtypes(include=["object"]).columns.tolist()
categorical_indices_all = [X_train_fs_raw.columns.get_loc(col) for col in categorical_features_all]

le_fs = LabelEncoder()
y_train_fs = le_fs.fit_transform(y_train_fs_text)
y_test_fs = le_fs.transform(y_test_fs_text)

# -----------------------------------------
# 7️⃣ موديل أولي لاختيار أفضل 30 سؤال
# -----------------------------------------

selector_model = CatBoostClassifier(
    iterations=250,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=100,
    class_weights=[1.0, 1.6, 1.0, 1.0]
)

selector_model.fit(
    X_train_fs_raw,
    y_train_fs,
    cat_features=categorical_indices_all,
    eval_set=(X_test_fs_raw, y_test_fs),
    use_best_model=True
)

feature_importances = pd.Series(
    selector_model.get_feature_importance(),
    index=X_train_fs_raw.columns
).sort_values(ascending=False)

TOP_K = 30
PRESELECT_K = 60

preselected_columns = feature_importances.head(PRESELECT_K).index.tolist()
selected_columns_ranked = preselected_columns[:TOP_K]

print("Selected original features (ranked):", len(selected_columns_ranked))
print(selected_columns_ranked)

X = X[selected_columns_ranked]

# -----------------------------------------
# 8️⃣ تجهيز Question Metadata
# -----------------------------------------

question_metadata = {}

for col in selected_columns_ranked:
    col_series = X[col]
    non_null = col_series.replace("Missing", np.nan).dropna()

    unique_values = sorted(non_null.unique().tolist()) if len(non_null) > 0 else []
    unique_count = len(unique_values)

    if pd.api.types.is_object_dtype(col_series):
        value_counts = non_null.astype(str).value_counts()
        top_options = value_counts.head(4).index.tolist()

        if len(value_counts) > 4 and "Other" not in top_options:
            top_options.append("Other")

        question_metadata[col] = {
            "question": col,
            "input_type": "select",
            "options": top_options
        }

    elif pd.api.types.is_numeric_dtype(col_series) and unique_count <= CATEGORICAL_NUMERIC_THRESHOLD:
        options = [str(v) for v in unique_values]

        if "Other" not in options:
            options.append("Other")

        question_metadata[col] = {
            "question": col,
            "input_type": "select",
            "options": options
        }

    else:
        question_metadata[col] = {
            "question": col,
            "input_type": "number",
            "options": [],
            "min": float(non_null.min()) if len(non_null) > 0 else None,
            "max": float(non_null.max()) if len(non_null) > 0 else None
        }

print("Question metadata prepared for:", len(question_metadata))

# -----------------------------------------
# 9️⃣ تقسيم الأسئلة إلى batches
# -----------------------------------------

batch_size = 10
question_batches = {}

num_batches = math.ceil(len(selected_columns_ranked) / batch_size)

for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    batch_name = f"batch_{i+1}"
    batch_features = selected_columns_ranked[start:end]

    question_batches[batch_name] = [
        question_metadata[col] for col in batch_features
    ]

print("Total batches:", len(question_batches))

# -----------------------------------------
# 10️⃣ تقسيم البيانات النهائي
# -----------------------------------------

X_train, X_test, y_train_text, y_test_text = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# خلي كل categorical نص ومفيش NaN
for col in X_train.select_dtypes(include=["object"]).columns:
    X_train[col] = X_train[col].fillna("Missing").astype(str)
    X_test[col] = X_test[col].fillna("Missing").astype(str)

categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_features]

# -----------------------------------------
# 11️⃣ ترميز الهدف النهائي
# -----------------------------------------

le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test = le.transform(y_test_text)

# -----------------------------------------
# 12️⃣ موديل CatBoost النهائي المحسّن
# -----------------------------------------

cat_model = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    l2_leaf_reg=7,
    random_strength=1.5,
    bagging_temperature=1.0,
    border_count=128,
    verbose=100,
    class_weights=[1.0, 1.6, 1.0, 1.0]
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# -----------------------------------------
# 13️⃣ تقييم الموديل
# -----------------------------------------

y_pred = cat_model.predict(X_test)
y_pred = np.array(y_pred).reshape(-1).astype(int)

print("\n📊 Improved CatBoost Model Performance Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------------------
# 14️⃣ تجهيز Best Answers
# -----------------------------------------

best_values = {}

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        non_null = X[col].dropna()
        best_values[col] = float(non_null.max()) if len(non_null) > 0 else 0.0
    else:
        mode_vals = X[col].replace("Missing", np.nan).dropna().mode()
        best_values[col] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "Other"

# -----------------------------------------
# 15️⃣ دالة التوقع للطالب
# -----------------------------------------

def predict_student(student_answers):
    total_features = len(selected_columns_ranked)
    answered = len(student_answers)

    full_input = best_values.copy()
    full_input.update(student_answers)

    student_df = pd.DataFrame([full_input])

    for col in selected_columns_ranked:
        if col not in student_df.columns:
            student_df[col] = best_values[col]

    student_df = student_df[selected_columns_ranked]

    # كل categorical لازم يكون string ومفيش NaN
    for col in student_df.select_dtypes(include=["object"]).columns:
        student_df[col] = student_df[col].fillna("Missing").astype(str)

    for col in student_df.columns:
        if col in converted_to_cat_cols:
            student_df[col] = student_df[col].apply(
                lambda x: str(int(round(float(x)))) if pd.notna(x) and str(x) != "Missing" else "Missing"
            )

    pred = cat_model.predict(student_df)
    pred_class = int(np.array(pred).reshape(-1)[0])

    pred_probs = cat_model.predict_proba(student_df)
    prob = float(np.max(pred_probs))

    label = le.inverse_transform([pred_class])[0]

    confidence = (answered / total_features) * prob

    return {
        "Predicted_Level": label,
        "Model_Probability": prob,
        "Prediction_Confidence": float(confidence)
    }

# -----------------------------------------
# 16️⃣ حفظ الملفات المهمة للـ API
# -----------------------------------------

cat_model.save_model("/kaggle/working/student_performance_model.cbm")
joblib.dump(le, "/kaggle/working/label_encoder.pkl")
joblib.dump(best_values, "/kaggle/working/best_values.pkl")
joblib.dump(selected_columns_ranked, "/kaggle/working/selected_features_ranked.pkl")
joblib.dump(converted_to_cat_cols, "/kaggle/working/converted_to_cat_cols.pkl")

with open("/kaggle/working/question_metadata.json", "w", encoding="utf-8") as f:
    json.dump(question_metadata, f, ensure_ascii=False, indent=2)

with open("/kaggle/working/question_batches.json", "w", encoding="utf-8") as f:
    json.dump(question_batches, f, ensure_ascii=False, indent=2)

print("\n✅ Saved files:")
print("/kaggle/working/student_performance_model.cbm")
print("/kaggle/working/label_encoder.pkl")
print("/kaggle/working/best_values.pkl")
print("/kaggle/working/selected_features_ranked.pkl")
print("/kaggle/working/converted_to_cat_cols.pkl")
print("/kaggle/working/question_metadata.json")
print("/kaggle/working/question_batches.json")
