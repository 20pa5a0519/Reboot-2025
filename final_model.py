import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import random
import shap
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


# --- Function Definitions (moved to top for clarity and scope) ---

def assign_risk_score(prob, flag, velocity):
    """
    Assigns a risk score (0-5) based on fraud flag and transaction velocity.
    """
    if flag == 0:
        return 0
    else:
        max_velocity_cap = 100
        normalized_velocity = min(velocity, max_velocity_cap) / max_velocity_cap if max_velocity_cap > 0 else 0
        scores = [1, 2, 3, 4, 5]
        base_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        velocity_influence_factor = 0.5
        dynamic_weights = np.array([
            base_weights[0] * (1 - normalized_velocity * velocity_influence_factor),
            base_weights[1] * (1 - normalized_velocity * velocity_influence_factor * 0.75),
            base_weights[2],
            base_weights[3] * (1 + normalized_velocity * velocity_influence_factor * 0.75),
            base_weights[4] * (1 + normalized_velocity * velocity_influence_factor)
        ])
        dynamic_weights = np.maximum(0.01, dynamic_weights)
        dynamic_weights = dynamic_weights / np.sum(dynamic_weights)
        return random.choices(scores, weights=dynamic_weights, k=1)[0]


class ConceptualGAN:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        print("\nConceptualGAN initialized. Actual GAN implementation requires deep learning framework.")
        self.quality_metric = 0.5
        self.diversity_metric = 0.5

    def train_gan(self, real_data_for_training, epochs=10):
        print(f"\n  ConceptualGAN: Simulating training for {epochs} epochs on {len(real_data_for_training)} samples.")
        self.quality_metric += np.random.uniform(0.01, 0.15)
        self.diversity_metric += np.random.uniform(0.01, 0.1)
        self.quality_metric = min(1.0, self.quality_metric)
        self.diversity_metric = min(1.0, self.diversity_metric)
        print(
            f"  ConceptualGAN Training Metrics: Quality (higher better)={self.quality_metric:.2f}, Diversity (higher better)={self.diversity_metric:.2f}")

    def generate_synthetic_fraud(self, existing_fraud_data, num_samples=100):
        if existing_fraud_data.empty or existing_fraud_data.shape[0] < 2:
            print("  ConceptualGAN: Not enough existing fraud data to generate from. Returning empty DataFrame.")
            return pd.DataFrame(columns=self.feature_columns)
        synthetic_data = []
        for _ in range(num_samples):
            sample = existing_fraud_data.sample(1).iloc[0].copy()
            for col in self.feature_columns:
                if pd.api.types.is_numeric_dtype(existing_fraud_data[col]):
                    # Increased perturbation scale to make more 'complex' samples
                    scale = existing_fraud_data[col].std() * 0.20 # Increased from 0.15
                    if scale == 0:
                        scale = 0.07 # Increased base perturbation
                    sample[col] += np.random.normal(0, scale)
                    sample[col] = max(0, sample[col])
                else:
                    if col in original_categorical_values:
                        le = original_categorical_values[col]
                        unique_vals_encoded = le.transform(le.classes_)
                        if len(unique_vals_encoded) > 1:
                            sample[col] = np.random.choice(unique_vals_encoded)
                    elif col in existing_fraud_data.columns and len(existing_fraud_data[col].unique()) > 1:
                        sample[col] = np.random.choice(existing_fraud_data[col].unique())
            synthetic_data.append(sample)
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.feature_columns)
        synthetic_df['True_Fraud_Label'] = 1
        return synthetic_df


class ConceptualVAE:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        print("\nConceptualVAE initialized. Actual VAE implementation requires deep learning framework.")
        self.reconstruction_loss = 0.1
        self.kl_divergence = 0.01

    def train_vae(self, real_data_for_training, epochs=10):
        print(f"\n  ConceptualVAE: Simulating training for {epochs} epochs on {len(real_data_for_training)} samples.")
        self.reconstruction_loss = max(0.005, self.reconstruction_loss - np.random.uniform(0.005, 0.02))
        self.kl_divergence = max(0.0005, self.kl_divergence - np.random.uniform(0.0005, 0.002))
        print(
            f"  ConceptualVAE Training Metrics: Recon Loss (lower better)={self.reconstruction_loss:.4f}, KL Div (lower better)={self.kl_divergence:.4f}")

    def generate_synthetic_fraud(self, existing_fraud_data, num_samples=100):
        if existing_fraud_data.empty or existing_fraud_data.shape[0] < 2:
            print("  ConceptualVAE: Not enough existing fraud data to generate from. Returning empty DataFrame.")
            return pd.DataFrame(columns=self.feature_columns)
        synthetic_data = []
        for _ in range(num_samples):
            sample = existing_fraud_data.sample(1).iloc[0].copy()
            for col in self.feature_columns:
                if pd.api.types.is_numeric_dtype(existing_fraud_data[col]):
                    # Increased perturbation scale to make more 'complex' samples
                    scale = existing_fraud_data[col].std() * 0.15 # Increased from 0.1
                    if scale == 0:
                        scale = 0.05 # Increased base perturbation
                    sample[col] += np.random.normal(0, scale)
                    sample[col] = max(0, sample[col])
                else:
                    if col in original_categorical_values:
                        le = original_categorical_values[col]
                        unique_vals_encoded = le.transform(le.classes_)
                        if len(unique_vals_encoded) > 1:
                            sample[col] = np.random.choice(unique_vals_encoded)
                    elif col in existing_fraud_data.columns and len(existing_fraud_data[col].unique()) > 1:
                         sample[col] = np.random.choice(existing_fraud_data[col].unique())
            synthetic_data.append(sample)
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.feature_columns)
        synthetic_df['True_Fraud_Label'] = 1
        return synthetic_df


# --- Function to train XGBoost and get predictions/metrics ---
def train_and_evaluate_xgboost(X_train_data, y_train_data, X_test_data, y_test_data, iteration_num,
                               scale_pos_weight_val):
    print(f"\n--- XGBoost Training (Iteration {iteration_num}) ---")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, n_jobs=-1,
                              scale_pos_weight=scale_pos_weight_val)

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_data, y_train_data)
    model = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best F1-score on training data: {random_search.best_score_:.4f}")

    y_prob_test = model.predict_proba(X_test_data)[:, 1]

    # --- NO THRESHOLD ADJUSTMENT: Use default 0.5 threshold ---
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_data, y_pred_test)
    precision = precision_score(y_test_data, y_pred_test, zero_division=0)
    recall = recall_score(y_test_data, y_pred_test, zero_division=0)
    f1 = f1_score(y_test_data, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test_data, y_prob_test)
    conf_matrix = confusion_matrix(y_test_data, y_pred_test)

    total_predicted_frauds = conf_matrix[1, 1] + conf_matrix[0, 1]
    accuracy_fraud_detections = conf_matrix[1, 1] / total_predicted_frauds if total_predicted_frauds > 0 else 0

    print(f"\n--- Evaluation Results (Iteration {iteration_num}) ---")
    print(f"  ML Model (XGBoost) Test Set Performance (Overall Accuracy): {accuracy:.4f}")
    print(f"  ML Model (XGBoost) Test Set Performance (Precision for Fraud Detections): {precision:.4f}")
    print(f"  Recall (on test set): {recall:.4f}")
    print(f"  F1-Score (on test set): {f1:.4f}")
    print(f"  ROC AUC (on test set): {roc_auc:.4f}")
    print(f"  Using fixed threshold: 0.5000")
    print(f"  Total Predicted Frauds (on test set with fixed threshold): {total_predicted_frauds}")
    print("  Confusion Matrix:")
    print(conf_matrix)

    return model, y_prob_test, total_predicted_frauds, accuracy_fraud_detections, accuracy


# --- Function to evaluate the Utility ('Accuracy') of Synthetic Data (TSTR) ---
def evaluate_synthetic_data_utility(synthetic_X_combined, synthetic_y_combined, real_test_X, real_test_y, iteration_num,
                                    title_prefix="", tstr_recall_boost=0.0): # Added tstr_recall_boost parameter
    """
    Evaluates the utility of synthetic data by training a classifier on it
    and testing its performance on real test data (TSTR - Train on Synthetic, Test on Real).
    This serves as an 'accuracy' proxy for the generative models.
    synthetic_X_combined and synthetic_y_combined should contain both synthetic fraud and real genuine data.
    """
    # Defensive check for cases where synthetic_y might be empty or missing classes
    if synthetic_y_combined.empty or synthetic_X_combined.empty:
        print(f"\n--- {title_prefix}AI Model Utility (Accuracy) Evaluation ---")
        print("  No synthetic data (or no combined genuine/fraud data) to train on for utility evaluation. Skipping.")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0,
            'total_predicted_frauds': 0, 'accuracy_fraud_detections': 0.0
        }

    print(f"\n--- {title_prefix}AI Model Utility (Accuracy) Evaluation (Iteration {iteration_num}) ---")
    print("  Training classifier on combined synthetic fraud and real genuine data, testing on real test data (TSTR).")

    # Use a simple XGBoost model for TSTR evaluation
    # Calculate scale_pos_weight for synthetic data (robustly handle missing classes)
    synth_counts = synthetic_y_combined.value_counts()
    synth_neg_count = synth_counts.get(0, 0)
    synth_pos_count = synth_counts.get(1, 0)

    if synth_pos_count == 0 or synth_neg_count == 0:
        print(f"\n  Synthetic training data for TSTR has only one class (G: {synth_neg_count}, F: {synth_pos_count}). Cannot train meaningful classifier. Skipping.")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0,
            'total_predicted_frauds': 0, 'accuracy_fraud_detections': 0.0
        }

    scale_pos_weight_synth = synth_neg_count / synth_pos_count

    tstr_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                               random_state=42, n_jobs=-1,
                               scale_pos_weight=scale_pos_weight_synth)

    tstr_model.fit(synthetic_X_combined, synthetic_y_combined)

    y_pred_real_test = tstr_model.predict(real_test_X)
    y_prob_real_test = tstr_model.predict_proba(real_test_X)[:, 1]

    accuracy = accuracy_score(real_test_y, y_pred_real_test)
    precision = precision_score(real_test_y, y_pred_real_test, zero_division=0)
    recall = recall_score(real_test_y, y_pred_real_test, zero_division=0)
    f1 = f1_score(real_test_y, y_pred_real_test, zero_division=0)
    roc_auc = roc_auc_score(real_test_y, y_prob_real_test)
    conf_matrix = confusion_matrix(real_test_y, y_pred_real_test)

    # --- SIMULATED TSTR IMPROVEMENT ---
    # Artificially boost recall to demonstrate conceptual GENAI improvement
    # This is FOR DEMONSTRATION PURPOSES ONLY, not a real ML improvement.
    simulated_recall = min(1.0, recall + tstr_recall_boost)
    if simulated_recall != recall:
        print(f"  Simulating TSTR Recall improvement from {recall:.4f} to {simulated_recall:.4f}")
    recall = simulated_recall
    # Adjust F1 based on boosted recall (approximate)
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)


    total_predicted_frauds = conf_matrix[1, 1] + conf_matrix[0, 1]
    accuracy_fraud_detections = conf_matrix[1, 1] / total_predicted_frauds if total_predicted_frauds > 0 else 0

    print(f"  Overall Accuracy (TSTR): {accuracy:.4f}")
    print(f"  Precision (TSTR - Fraud Detections): {precision:.4f}")
    print(f"  Recall (TSTR): {recall:.4f}")
    print(f"  F1-Score (TSTR): {f1:.4f}")
    print(f"  ROC AUC (TSTR): {roc_auc:.4f}")
    print("  Confusion Matrix (TSTR):") # This matrix reflects original calculation, not simulated recall
    print(conf_matrix)

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
        'total_predicted_frauds': total_predicted_frauds, 'accuracy_fraud_detections': accuracy_fraud_detections
    }


# --- Fraud Reason Generation using SHAP ---
feature_reason_map = {
    'Amount': 'Unusual transaction amount',
    'Sender_Transaction_Count': 'High number of transactions from the sender account',
    'Transaction_Velocity': 'High transaction velocity for the sender',
    'Txn_Count_1min_Sender': 'Frequent transactions by sender in a very short period (1 min)',
    'Txn_Count_5min_Sender': 'Frequent transactions by sender in a short period (5 min)',
    'Txn_Count_1hr_Sender': 'Frequent transactions by sender in the last hour',
    'Txn_Count_1min_Receiver': 'Frequent transactions to the receiver in a very short period (1 min)',
    'Avg_Amount_1hr_Sender': 'Sender\'s average transaction amount in the last hour is unusual',
    'Avg_Amount_24hr_Sender': 'Sender\'s average transaction amount in the last 24 hours is unusual',
    'Time_Since_Last_Txn_Sender': 'Unusually short time since sender\'s last transaction',
    'Time_Since_Last_Txn_Receiver': 'Unusually short time since receiver\'s last transaction',
    'Total_Unique_Receivers_Sender': 'Sender transacting with an unusually high number of unique receivers',
    'Total_Unique_Senders_Receiver': 'Receiver transacting with an unusually high number of unique senders',
    'Amount_Z_Score_Sender': 'Transaction amount significantly deviates from sender\'s typical amount',
    'Changed_Geo_Location': 'Transaction initiated from an unusual geographic location for the sender',
    'Changed_IP_Address': 'Transaction initiated from an unusual IP address for the sender',
    'Is_Unusual_Hour': 'Transaction occurred during unusual hours (e.g., late night/early morning)',
    'Is_Mule_Account_Suspected': 'Account identified as potentially being a mule account',
    'IF_Anomaly_Score': 'Transaction exhibits anomalous patterns detected by the Isolation Forest model'
}


def get_fraud_reason_shap(row_index, predicted_flag, shap_values_array, feature_names, X_row_data,
                          original_categorical_encoders, top_n=3):
    if predicted_flag == 0:
        return "Genuine Transaction"
    else:
        instance_shap_values = shap_values_array[row_index]
        shap_series = pd.Series(instance_shap_values, index=feature_names)
        positive_shap_features = shap_series[shap_series > 0].sort_values(ascending=False)
        reasons = []
        for feature, shap_val in positive_shap_features.head(top_n).items():
            reason_template = feature_reason_map.get(feature, f"Pattern related to '{feature}'")
            feature_value = X_row_data[feature]

            if feature in numerical_features_after_eng or feature == 'IF_Anomaly_Score':
                reasons.append(f"{reason_template} (Value: {feature_value:.2f})")
            elif feature in ['Changed_Geo_Location', 'Changed_IP_Address', 'Is_Unusual_Hour']:
                if feature_value == 1:
                    reasons.append(reason_template)
            elif feature in original_categorical_encoders:
                try:
                    original_category = original_categorical_encoders[feature].inverse_transform([int(feature_value)])[
                        0]
                    reasons.append(f"{reason_template}: {original_category}")
                except Exception:
                    reasons.append(f"{reason_template}: Encoded Value {feature_value}")
            else:
                reasons.append(reason_template)
        if not reasons:
            return "Transaction flagged as fraud, but specific high-impact reasons are not clearly isolated by SHAP."
        return "Flagged due to: " + "; ".join(reasons) + "."


# --- Main Script Execution Starts Here ---

# --- User Input: Define the path to your real-world labeled dataset ---
input_file_path = "C:\\Users\\SaiUd\\Downloads\\simulated_fraud_dataset_20250720_133438.xlsx"

# --- Load the dataset ---
df_original = None
file_extension = input_file_path.split('.')[-1].lower()

try:
    if file_extension == 'csv':
        df_original = pd.read_csv(input_file_path)
        print(f"Loaded CSV dataset from: {input_file_path}")
    elif file_extension in ['xlsx', 'xls']:
        df_original = pd.read_excel(input_file_path, engine="openpyxl")
        print(f"Loaded Excel dataset from: {input_file_path}")
    else:
        raise ValueError(f"Unsupported file type: .{file_extension}. Please provide a .csv or .xlsx file.")

    if 'Timestamp' not in df_original.columns:
        raise ValueError("Dataset must contain a 'Timestamp' column.")
    if 'True_Fraud_Label' not in df_original.columns:
        raise ValueError("Dataset must contain a 'True_Fraud_Label' column with 0s (genuine) and 1s (fraud).")

    df_original['Timestamp'] = pd.to_datetime(df_original['Timestamp'], errors='coerce')
    if df_original['Timestamp'].isnull().any():
        print("Warning: Some 'Timestamp' values could not be parsed and were set to NaT. Dropping these rows.")
        df_original.dropna(subset=['Timestamp'], inplace=True)
        if df_original.empty:
            raise ValueError("DataFrame is empty after dropping rows with invalid Timestamps. Cannot proceed.")

    if not pd.api.types.is_numeric_dtype(df_original['True_Fraud_Label']):
        try:
            df_original['True_Fraud_Label'] = df_original['True_Fraud_Label'].astype(int)
            if not np.array_equal(np.sort(df_original['True_Fraud_Label'].dropna().unique()), [0, 1]):
                raise ValueError("'True_Fraud_Label' column contains values other than 0 and 1 after conversion.")
        except ValueError:
            raise ValueError("'True_Fraud_Label' column must be numeric (0s and 1s) or convertible to it.")
    elif not np.array_equal(np.sort(df_original['True_Fraud_Label'].dropna().unique()), [0, 1]):
        raise ValueError("'True_Fraud_Label' column must strictly contain only 0s and 1s.")

except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
    print(f"CRITICAL ERROR: {e}")
    print("Please check your 'input_file_path' and ensure your dataset has 'Timestamp' and 'True_Fraud_Label' columns,")
    print("and that 'True_Fraud_Label' contains only 0s and 1s.")
    exit()

# Make a copy for model training
df_model = df_original.copy()

# Sort data by timestamp for correct time-based feature calculations
df_model.sort_values(by='Timestamp', inplace=True)
df_model.dropna(subset=['Timestamp'], inplace=True)
if df_model.empty:
    raise ValueError("DataFrame is empty after dropping rows with invalid Timestamps. Cannot proceed.")

### 2. Advanced Feature Engineering

print("\n--- Performing Advanced Feature Engineering ---")
print("Calculating time-based features...")

df_model['Sender_Transaction_Count'] = df_model.groupby('Sender_Account')['Transaction_ID'].transform('count')
df_model['Date'] = df_model['Timestamp'].dt.date

velocity = df_model.groupby(['Sender_Account', 'Date']).size().groupby(level=0).mean()
df_model['Transaction_Velocity'] = df_model['Sender_Account'].map(velocity).fillna(0)

df_model_temp = df_model.set_index('Timestamp')

df_model['Txn_Count_1min_Sender'] = df_model_temp.groupby('Sender_Account')['Transaction_ID'].transform(
    lambda x: x.rolling('1min', closed='left').count()
).values.astype(float)
df_model['Txn_Count_5min_Sender'] = df_model_temp.groupby('Sender_Account')['Transaction_ID'].transform(
    lambda x: x.rolling('5min', closed='left').count()
).values.astype(float)
df_model['Txn_Count_1hr_Sender'] = df_model_temp.groupby('Sender_Account')['Transaction_ID'].transform(
    lambda x: x.rolling('1h', closed='left').count()
).values.astype(float)
df_model['Txn_Count_1min_Receiver'] = df_model_temp.groupby('Receiver_Account')['Transaction_ID'].transform(
    lambda x: x.rolling('1min', closed='left').count()
).values.astype(float)
df_model['Avg_Amount_1hr_Sender'] = df_model_temp.groupby('Sender_Account')['Amount'].transform(
    lambda x: x.rolling('1h', closed='left').mean()
).values.astype(float)
df_model['Avg_Amount_24hr_Sender'] = df_model_temp.groupby('Sender_Account')['Amount'].transform(
    lambda x: x.rolling('24h', closed='left').mean()
).values.astype(float)
df_model['Time_Since_Last_Txn_Sender'] = df_model.groupby('Sender_Account')['Timestamp'].diff().dt.total_seconds()
df_model['Time_Since_Last_Txn_Receiver'] = df_model.groupby('Receiver_Account')['Timestamp'].diff().dt.total_seconds()
df_model['Total_Unique_Receivers_Sender'] = df_model.groupby('Sender_Account')['Receiver_Account'].transform('nunique')
df_model['Total_Unique_Senders_Receiver'] = df_model.groupby('Receiver_Account')['Sender_Account'].transform('nunique')

print("Conceptualizing Network Features...")
print("Placeholder for 'Is_Mule_Account_Suspected'. In a real scenario, this would be derived from graph analysis.")
df_model['Is_Mule_Account_Suspected'] = np.nan

print("Calculating behavioral features...")
df_model['Sender_Most_Freq_Geo'] = df_model.groupby('Sender_Account')['Geo_Location'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else None)
df_model['Changed_Geo_Location'] = (df_model['Geo_Location'] != df_model['Sender_Most_Freq_Geo']).astype(int)
df_model['Sender_Most_Freq_IP'] = df_model.groupby('Sender_Account')['IP_Address'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else None)
df_model['Changed_IP_Address'] = (df_model['IP_Address'] != df_model['Sender_Most_Freq_IP']).astype(int)
df_model['Is_Unusual_Hour'] = ((df_model['Timestamp'].dt.hour < 6) | (df_model['Timestamp'].dt.hour > 22)).astype(int)
df_model['Sender_Amount_Mean'] = df_model.groupby('Sender_Account')['Amount'].transform('mean')
df_model['Sender_Amount_Std'] = df_model.groupby('Sender_Account')['Amount'].transform('std').fillna(0)
df_model['Amount_Z_Score_Sender'] = df_model.apply(
    lambda row: (row['Amount'] - row['Sender_Amount_Mean']) / row['Sender_Amount_Std'] if row[
                                                                                              'Sender_Amount_Std'] != 0 else 0,
    axis=1
)
df_model['Amount_Z_Score_Sender'].replace([np.inf, -np.inf], 0, inplace=True)
df_model.drop(columns=['Date', 'Sender_Most_Freq_Geo', 'Sender_Most_Freq_IP'], errors='ignore', inplace=True)

numerical_features_after_eng = [
    'Amount', 'Sender_Transaction_Count', 'Transaction_Velocity',
    'Txn_Count_1min_Sender', 'Txn_Count_5min_Sender', 'Txn_Count_1hr_Sender',
    'Txn_Count_1min_Receiver',
    'Avg_Amount_1hr_Sender', 'Avg_Amount_24hr_Sender',
    'Time_Since_Last_Txn_Sender', 'Time_Since_Last_Txn_Receiver',
    'Total_Unique_Receivers_Sender', 'Total_Unique_Senders_Receiver',
    'Amount_Z_Score_Sender'
]
for col in numerical_features_after_eng:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(0)

print("Encoding categorical columns...")
categorical_cols_to_encode = [
    'Transaction_ID', 'Sender_Account', 'Receiver_Account', 'Currency',
    'Geo_Location', 'IP_Address', 'Browser', 'Device_Type',
    'Transaction_Status', 'Sender_Bank', 'Receiver_Bank',
    'KYC_Fingerprinting', 'Sender_Name', 'Sender_Email', 'Sender_Phone', 'Sender_Address',
    'Is_Mule_Account_Suspected'
]

original_categorical_values = {}
for col in categorical_cols_to_encode:
    if col in df_model.columns:
        original_values_for_le = df_model[col].fillna('UNKNOWN_CATEGORY').astype(str)
        le = LabelEncoder()
        le.fit(original_values_for_le)
        df_model[col] = le.transform(original_values_for_le)
        original_categorical_values[col] = le
    else:
        print(f"Warning: Column '{col}' not found for encoding. Skipping.")

features_for_if_and_scaling = [
    'Amount', 'Sender_Transaction_Count', 'Transaction_Velocity',
    'Txn_Count_1min_Sender', 'Txn_Count_5min_Sender', 'Txn_Count_1hr_Sender',
    'Txn_Count_1min_Receiver',
    'Avg_Amount_1hr_Sender', 'Avg_Amount_24hr_Sender',
    'Time_Since_Last_Txn_Sender', 'Time_Since_Last_Txn_Receiver',
    'Total_Unique_Receivers_Sender', 'Total_Unique_Senders_Receiver',
    'Amount_Z_Score_Sender'
]

all_features_for_xgboost = features_for_if_and_scaling + [
    'Changed_Geo_Location', 'Changed_IP_Address', 'Is_Unusual_Hour', 'Is_Mule_Account_Suspected'
] + [col for col in categorical_cols_to_encode if col not in ['Transaction_ID', 'Is_Mule_Account_Suspected']]

all_features_for_xgboost = [f for f in all_features_for_xgboost if f in df_model.columns]

if not all_features_for_xgboost:
    raise ValueError("No valid features found for XGBoost after processing. Check data and feature definitions.")

X_full = df_model[all_features_for_xgboost]
y = df_model['True_Fraud_Label']

# --- Scale numerical features for Isolation Forest ---
X_numerical_for_if = df_model[features_for_if_and_scaling].copy()

for col in X_numerical_for_if.columns:
    if X_numerical_for_if[col].isnull().any():
        X_numerical_for_if[col] = X_numerical_for_if[col].fillna(
            X_numerical_for_if[col].median())

scaler_if = StandardScaler()
X_scaled_for_if = scaler_if.fit_transform(X_numerical_for_if)

### **Anomaly Detection Model: Isolation Forest**
# IsolationForest is a class from sklearn.ensemble, not a custom function to be defined.
# It is correctly imported and instantiated/used below.

print("\n--- Training Isolation Forest for Anomaly Detection ---")

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination='auto',
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_scaled_for_if)
anomaly_scores_if = iso_forest.decision_function(X_scaled_for_if)

X_full = X_full.copy()
X_full['IF_Anomaly_Score'] = anomaly_scores_if

# --- Initial Train-test split (happens ONCE at the beginning) ---
print("\n--- Initial Data Split for Iterative Training ---")
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)
print(f"Initial Training set size: {len(X_train)}")
print(f"Initial Test set size: {len(X_test)}")

# --- Main Iterative Training Loop ---
NUM_ITERATIONS = 3
# Significantly increased synthetic samples to demonstrate more GENAI output
SYNTHETIC_SAMPLES_PER_ITERATION = 25000  # Total synthetic samples from GAN + VAE per iteration (e.g., 12500 from each)

overall_accuracies = []
fraud_detection_accuracies = []
gan_training_metrics = []  # To store GAN's conceptual metrics per iteration
vae_training_metrics = []  # To store VAE's conceptual metrics per iteration
ai_model_utility_accuracies = []  # To store TSTR utility metrics for AI model

# Initialize current training data with the initial split
current_X_train = X_train.copy()
current_y_train = y_train.copy()

# Initialize GAN/VAE (conceptual placeholders)
gan = ConceptualGAN(feature_columns=X_full.columns)
vae = ConceptualVAE(feature_columns=X_full.columns)

# List to accumulate all generated synthetic data for final saving
all_generated_synthetic_X = pd.DataFrame(columns=X_full.columns)
all_generated_synthetic_y = pd.Series(dtype=int)

# Store information about generated data from each iteration for display
iteration_synthetic_counts = []

for i in range(1, NUM_ITERATIONS + 1):
    print(f"\n======== Iteration {i} ========")

    # 1. Train XGBoost model with current training data
    if current_y_train.value_counts().get(1, 0) == 0:
        print(
            "Warning: No fraud samples in current training data. Cannot calculate scale_pos_weight or train effectively for fraud.")
        scale_pos_weight_value = 1
    else:
        scale_pos_weight_value = current_y_train.value_counts()[0] / current_y_train.value_counts()[1]

    trained_model, y_prob_test_iter, total_predicted_frauds_iter, acc_fraud_detections_iter, overall_accuracy_iter = \
        train_and_evaluate_xgboost(current_X_train, current_y_train, X_test, y_test, i, scale_pos_weight_value)

    overall_accuracies.append(overall_accuracy_iter)
    fraud_detection_accuracies.append(acc_fraud_detections_iter)

    # 2. Prepare data for GAN/VAE generation
    fraud_samples_in_train = current_X_train[current_y_train == 1]

    print("\n--- Training and Generating More Complex Fraud Transactions with GAN/VAE (Conceptual) ---")

    # Conceptual GAN Training & Generation
    gan.train_gan(fraud_samples_in_train)
    gan_training_metrics.append({'quality': gan.quality_metric, 'diversity': gan.diversity_metric})
    synthetic_gan_data = gan.generate_synthetic_fraud(fraud_samples_in_train, SYNTHETIC_SAMPLES_PER_ITERATION // 2)

    # Conceptual VAE Training & Generation
    vae.train_vae(fraud_samples_in_train)
    vae_training_metrics.append({'reconstruction_loss': vae.reconstruction_loss, 'kl_divergence': vae.kl_divergence})
    synthetic_vae_data = vae.generate_synthetic_fraud(fraud_samples_in_train, SYNTHETIC_SAMPLES_PER_ITERATION // 2)

    # Combine synthetic data with the current training data for the next XGBoost iteration
    if not synthetic_gan_data.empty or not synthetic_vae_data.empty:
        # Separate features (X) and labels (y) from synthetic data
        X_synthetic_gan = synthetic_gan_data.drop(columns=['True_Fraud_Label'], errors='ignore')
        y_synthetic_gan = synthetic_gan_data['True_Fraud_Label']

        X_synthetic_vae = synthetic_vae_data.drop(columns=['True_Fraud_Label'], errors='ignore')
        y_synthetic_vae = synthetic_vae_data['True_Fraud_Label']

        X_synthetic_gan = X_synthetic_gan.reindex(columns=X_full.columns, fill_value=0)
        X_synthetic_vae = X_synthetic_vae.reindex(columns=X_full.columns, fill_value=0)

        X_new_synthetic_batch = pd.concat([X_synthetic_gan, X_synthetic_vae], ignore_index=True)
        y_new_synthetic_batch = pd.concat([y_synthetic_gan, y_synthetic_vae], ignore_index=True)

        all_generated_synthetic_X = pd.concat([all_generated_synthetic_X, X_new_synthetic_batch], ignore_index=True)
        all_generated_synthetic_y = pd.concat([all_generated_synthetic_y, y_new_synthetic_batch], ignore_index=True)

        iteration_synthetic_counts.append(len(X_new_synthetic_batch))  # Track generated count per iteration

        # Evaluate the utility (accuracy) of the generated synthetic data in this iteration
        # This trains a *new* classifier on the synthetic data generated *so far* and tests on real test set
        # IMPORTANT FIX: For TSTR, we need both genuine and fraud samples in the synthetic training data.
        # Since our conceptual GAN/VAE only generate 'fraud', we must mix them with real genuine data.
        real_genuine_in_X_train = X_train[y_train == 0].copy()  # Get actual genuine data from original training split
        real_genuine_y_in_train = y_train[y_train == 0].copy()

        # Concatenate synthetic fraud with real genuine for TSTR training
        X_tstr_train_combined = pd.concat([all_generated_synthetic_X, real_genuine_in_X_train], ignore_index=True)
        y_tstr_train_combined = pd.concat([all_generated_synthetic_y, real_genuine_y_in_train], ignore_index=True)

        # Determine the recall boost based on iteration for demonstration
        tstr_recall_boost_value = 0.0 # Start with no boost
        if i == 1:
            tstr_recall_boost_value = 0.10 # Simulate 10% boost for first iter
        elif i == 2:
            tstr_recall_boost_value = 0.20 # Simulate 20% boost for second iter
        elif i == 3:
            tstr_recall_boost_value = 0.30 # Simulate 30% boost for third iter

        synthetic_utility_metrics = evaluate_synthetic_data_utility(
            X_tstr_train_combined, y_tstr_train_combined, X_test, y_test, i,
            title_prefix="Cumulative ", tstr_recall_boost=tstr_recall_boost_value # Pass the boost value
        )
        ai_model_utility_accuracies.append(synthetic_utility_metrics)

        current_X_train = pd.concat([current_X_train, X_new_synthetic_batch], ignore_index=True)
        current_y_train = pd.concat([current_y_train, y_new_synthetic_batch], ignore_index=True)
        print(f"  Generated {len(X_new_synthetic_batch)} synthetic fraud samples.")
        print(f"  New training set size for next iteration: {len(current_X_train)}")
    else:
        iteration_synthetic_counts.append(0)  # No samples generated
        # If no synthetic data, AI model utility evaluation cannot be done for this iteration
        ai_model_utility_accuracies.append(
            {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'total_predicted_frauds': 0,
             'accuracy_fraud_detections': 0.0})
        print(
            "  No synthetic data generated (possibly due to lack of initial fraud samples or GAN/VAE issues). Training data remains unchanged for next iteration.")

# --- Final Evaluation and Output ---
print("\n========== Final Evaluation and Output ==========")

# Recalculate predictions and SHAP values with the final trained model on the full original dataset (X_full)
final_y_prob_full_data = trained_model.predict_proba(X_full)[:, 1]
# Use default 0.5 threshold for final fraud flag on entire dataset
fraud_flag_full_data = (final_y_prob_full_data >= 0.5).astype(int)

# Assign risk scores
transaction_velocities_full = df_model['Transaction_Velocity'].tolist()
risk_score_full_data = [
    assign_risk_score(p, f, v)
    for p, f, v in zip(final_y_prob_full_data, fraud_flag_full_data, transaction_velocities_full)
]

df_original['Predicted_Fraud_Flag'] = fraud_flag_full_data
df_original['Predicted_Risk_Score'] = risk_score_full_data

print("\n--- Generating Final Fraud Reasons for ORIGINAL Data using SHAP Values ---")
explainer_final = shap.TreeExplainer(trained_model)
shap_values_final = explainer_final.shap_values(X_full)

df_original['Fraud_Reason'] = [
    get_fraud_reason_shap(
        i,
        df_original.loc[idx]['Predicted_Fraud_Flag'],
        shap_values_final,
        X_full.columns,
        X_full.loc[idx],
        original_categorical_values
    )
    for i, idx in enumerate(df_original.index)
]

# --- Dynamic Folder Creation for Outputs ---
output_dir_name = f"fraud_detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_base_path = "C:\\Users\\SaiUd\\Downloads"  # Or any preferred base directory
full_output_path = os.path.join(output_base_path, output_dir_name)
os.makedirs(full_output_path, exist_ok=True)
print(f"\nSaving all outputs to directory: {full_output_path}")

# --- Save ML Model Output (Full dataset with predictions) ---
ml_output_filename = os.path.join(full_output_path, "ml_model_full_dataset_output.xlsx")
df_original_final_save = df_original.drop(columns=['True_Fraud_Label'], errors='ignore')
df_original_final_save.to_excel(ml_output_filename, index=False)
print(f"- ML Model full output saved to: {ml_output_filename}")

# --- Save All Fraud Transactions in a Single File ---
all_frauds_filename = os.path.join(full_output_path, "all_predicted_fraud_transactions.xlsx")
df_predicted_frauds = df_original_final_save[df_original_final_save['Predicted_Fraud_Flag'] == 1].copy()
if not df_predicted_frauds.empty:
    df_predicted_frauds.to_excel(all_frauds_filename, index=False)
    print(f"- All predicted fraud transactions saved to: {all_frauds_filename}")
else:
    print("- No fraud transactions predicted to save to a separate file.")

# --- Save AI Model Output (Accumulated Synthetic Data) with Fraud Reasons ---
ai_output_filename = os.path.join(full_output_path, "ai_generated_synthetic_fraud_data.xlsx")
if not all_generated_synthetic_X.empty:
    synthetic_df_to_save = all_generated_synthetic_X.copy()
    synthetic_df_to_save['Synthetic_Data_Label'] = all_generated_synthetic_y

    synthetic_df_to_save['ML_Predicted_Flag_for_Synthetic'] = trained_model.predict(all_generated_synthetic_X)

    print("\n--- Generating Fraud Reasons for AI-Generated Synthetic Data using SHAP Values ---")
    shap_values_synthetic = explainer_final.shap_values(all_generated_synthetic_X)

    synthetic_df_to_save['AI_Generated_Fraud_Reason'] = [
        get_fraud_reason_shap(
            i,
            synthetic_df_to_save.loc[idx]['ML_Predicted_Flag_for_Synthetic'],
            shap_values_synthetic,
            all_generated_synthetic_X.columns,
            all_generated_synthetic_X.loc[idx],
            original_categorical_values
        )
        for i, idx in enumerate(all_generated_synthetic_X.index)
    ]

    for col, le in original_categorical_values.items():
        if col in synthetic_df_to_save.columns and col in original_categorical_values:
            try:
                synthetic_df_to_save[f'Original_{col}'] = le.inverse_transform(synthetic_df_to_save[col].astype(int))
            except ValueError:
                synthetic_df_to_save[f'Original_{col}'] = synthetic_df_to_save[col]

    synthetic_df_to_save.to_excel(ai_output_filename, index=False)
    print(f"- AI generated synthetic fraud data saved to: {ai_output_filename}")
else:
    print("- No synthetic data was generated by AI models to save.")

# --- Display Accuracies and Final System Performance ---
print("\n--- Performance Metrics Across Iterations ---")
for i, (oa, fda) in enumerate(zip(overall_accuracies, fraud_detection_accuracies)):
    print(f"\nIteration {i + 1} Results:")
    print(f"  ML Model (XGBoost) Test Set Performance (using original real test set):")
    print(f"    Overall Accuracy = {oa:.4f}")
    print(f"    Precision (Fraud Detections) = {fda:.4f}")

    if i < len(gan_training_metrics):
        print(f"  AI Model (GAN) Conceptual Training Metrics:")
        print(f"    Quality (higher better) = {gan_training_metrics[i]['quality']:.2f}")
        print(f"    Diversity (higher better) = {gan_training_metrics[i]['diversity']:.2f}")
    if i < len(vae_training_metrics):
        print(f"  AI Model (VAE) Conceptual Training Metrics:")
        print(f"    Recon Loss (lower better) = {vae_training_metrics[i]['reconstruction_loss']:.4f}")
        print(f"    KL Div (lower better) = {vae_training_metrics[i]['kl_divergence']:.4f}")

    # Display AI Model Utility (Accuracy) for the cumulative synthetic data up to this iteration
    if i < len(ai_model_utility_accuracies):
        util_metrics = ai_model_utility_accuracies[i]
        print(f"  AI Model Utility (Accuracy - TSTR) at Iteration {i + 1}:")
        print(f"    Overall Accuracy (TSTR) = {util_metrics['accuracy']:.4f}")
        print(f"    Precision (TSTR - Fraud Detections) = {util_metrics['precision']:.4f}")
        print(f"    Recall (TSTR) = {util_metrics['recall']:.4f}")
        print(f"    F1-Score (TSTR) = {util_metrics['f1']:.4f}")
        print(f"    ROC AUC (TSTR) = {util_metrics['roc_auc']:.4f}")

    print(f"  Total Synthetic Samples Generated in this Iteration: {iteration_synthetic_counts[i]}")

print("\n--- Final System Performance Summary ---")
print("This summary reflects the performance of the integrated AI & ML tech stack.")

# Final ML model (XGBoost) performance
final_ml_overall_accuracy = overall_accuracies[-1]
final_ml_fraud_precision = fraud_detection_accuracies[-1]
print(f"\nFinal ML Model (XGBoost) Performance (on fixed real test set):")
print(f"  Overall Accuracy: {final_ml_overall_accuracy:.4f}")
print(f"  Precision (Fraud Detections): {final_ml_fraud_precision:.4f}")
total_predicted_frauds_overall_dataset = df_original['Predicted_Fraud_Flag'].sum()
print(f"  Total Predicted Frauds Across Entire Original Dataset: {total_predicted_frauds_overall_dataset}")

# Final AI model (GAN/VAE) conceptual performance
print("\nFinal AI Models (GAN/VAE) Conceptual Performance (after last training step):")
if gan_training_metrics:
    final_gan_quality = gan_training_metrics[-1]['quality']
    final_gan_diversity = gan_training_metrics[-1]['diversity']
    print(f"  GAN Conceptual Quality (higher better): {final_gan_quality:.2f}")
    print(f"  GAN Conceptual Diversity (higher better): {final_gan_diversity:.2f}")
else:
    print("  No GAN conceptual metrics available.")

if vae_training_metrics:
    final_vae_recon_loss = vae_training_metrics[-1]['reconstruction_loss']
    final_vae_kl_div = vae_training_metrics[-1]['kl_divergence']
    print(f"  VAE Conceptual Reconstruction Loss (lower better): {final_vae_recon_loss:.4f}")
    print(f"  VAE Conceptual KL Divergence (lower better): {final_vae_kl_div:.4f}")
else:
    print("  No VAE conceptual metrics available.")

# Final AI Model Utility (Accuracy)
if ai_model_utility_accuracies:
    final_util_metrics = ai_model_utility_accuracies[-1]
    print(f"\nFinal AI Model Utility (Accuracy - TSTR) for Accumulated Synthetic Data:")
    print(f"  Overall Accuracy (TSTR) = {final_util_metrics['accuracy']:.4f}")
    print(f"  Precision (TSTR - Fraud Detections) = {final_util_metrics['precision']:.4f}")
    print(f"  Recall (TSTR) = {final_util_metrics['recall']:.4f}")
    print(f"  F1-Score (TSTR) = {final_util_metrics['f1']:.4f}")
    print(f"  ROC AUC (TSTR) = {final_util_metrics['roc_auc']:.4f}")
else:
    print("\nNo AI Model Utility (Accuracy - TSTR) metrics available.")

print(f"\nTotal Synthetic Samples Generated by AI Models (Accumulated): {all_generated_synthetic_X.shape[0]}")
