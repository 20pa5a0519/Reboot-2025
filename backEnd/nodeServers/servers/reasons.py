import pandas as pd

# Load the dataset
df = pd.read_csv("hope_final.csv")

# Filter transactions that are flagged as fraud
flagged_df = df[df['Predicted_Fraud_Flag'] == 1]

# Store results
results = []

# Define reason generation function
def generate_reasons(account_df):
    reasons = []

    # 1. Check for high-value failed or pending transactions
    suspicious_status = account_df[
        (account_df['Transaction_Status'].isin(['Failed', 'Pending'])) &
        (account_df['Amount'] > account_df['Amount'].mean())
    ]
    if not suspicious_status.empty:
        reasons.append("High-value transaction with failed or pending status.")

    # 2. Check for multiple currencies used
    if account_df['Currency'].nunique() > 1:
        reasons.append("Multiple currencies used in transactions.")

    # 3. Check for uncommon browsers
    uncommon_browsers = ['Comodo Dragon', 'Waterfox', 'DarkFox', 'Brave']
    if any(account_df['Browser'].isin(uncommon_browsers)):
        reasons.append("Uncommon browser used, possibly hiding identity.")

    # 4. Check if sender used both mobile and desktop
    if account_df['Device_Type'].nunique() > 1:
        reasons.append("Transactions from multiple device types (mobile/desktop).")

    # 5. Check for high number of receivers
    if account_df['Receiver_Account'].nunique() > 3:
        reasons.append("Multiple unique receiver accounts in short span.")

    # Return only top 3 reasons
    return reasons[:3]

# Group by sender account and generate reasons
for account_id, group in flagged_df.groupby('Sender_Account'):
    reasons = generate_reasons(group)
    while len(reasons) < 3:
        reasons.append("No further suspicious pattern detected.")
    results.append({
        'Sender_Account': account_id,
        'Reason_1': reasons[0],
        'Reason_2': reasons[1],
        'Reason_3': reasons[2],
    })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

print("✅ Fraud reason analysis saved to results.csv")
