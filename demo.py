import pandas as pd
from neo4j import GraphDatabase, basic_auth
import logging
import time
import traceback # Import traceback for detailed error logging

# Configure logging for Neo4j driver
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Neo4j Connection Details ---
# IMPORTANT: Ensure your Neo4j database is running and accessible at this URI
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "Genaifraud" # Default password, change if you've set a new one

# --- 2. Load Data from Excel ---
excel_file_path = 'E:/Downloads/hope_final.xlsx'
# IMPORTANT: Sheet name corrected to 'hope_final' as per your clarification
sheet_name = 'hope_final' 

# Initialize variables outside the try block to ensure they are always defined
df = pd.DataFrame() # Initialize an empty DataFrame

driver = None # Initialize driver outside try block

try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    logging.info(f"Successfully loaded data from '{excel_file_path}', sheet '{sheet_name}'. Shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")

    driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
    
    # Add a small delay to ensure the database is fully ready, if needed
    # time.sleep(2) # Uncomment this line if connection still fails immediately

    driver.verify_connectivity()
    logging.info("Connected to Neo4j database successfully!")

    with driver.session() as session:
        # Optional: Clear existing data for a clean import
        logging.info("Clearing existing data in Neo4j (optional step)...")
        session.run("MATCH (n) DETACH DELETE n")
        logging.info("Existing data cleared.")

        # Create unique constraints for Account nodes for performance
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
        logging.info("Created unique constraint for Account nodes.")

        # Prepare properties for account nodes from the DataFrame
        # We'll collect all relevant account properties from both sender and receiver sides
        # And specifically prioritize Predicted_Fraud_Flag and Predicted_Risk_Score
        account_properties = {}

        for index, row in df.iterrows():
            sender_id = str(row['Sender_Account'])
            receiver_id = str(row['Receiver_Account'])

            # Initialize account_properties for sender if not exists
            if sender_id not in account_properties:
                account_properties[sender_id] = {'account_id': sender_id}
            
            # Initialize account_properties for receiver if not exists
            if receiver_id not in account_properties:
                account_properties[receiver_id] = {'account_id': receiver_id}

            # Update sender properties based on transaction data
            # Prioritize fraud flags/risk scores
            current_sender_fraud_flag = account_properties[sender_id].get('Predicted_Fraud_Flag', 0)
            current_sender_risk_score = account_properties[sender_id].get('Predicted_Risk_Score', 0.0)

            if pd.notna(row.get('Predicted_Fraud_Flag')):
                new_fraud_flag = int(row['Predicted_Fraud_Flag'])
                if new_fraud_flag > current_sender_fraud_flag: # Take the higher fraud flag
                    account_properties[sender_id]['Predicted_Fraud_Flag'] = new_fraud_flag
            
            if pd.notna(row.get('Predicted_Risk_Score')):
                new_risk_score = float(row['Predicted_Risk_Score'])
                if new_risk_score > current_sender_risk_score: # Take the higher risk score
                    account_properties[sender_id]['Predicted_Risk_Score'] = new_risk_score

            # Add other account-specific properties if they exist in your Excel for sender
            if pd.notna(row.get('Sender_Account_Holder')):
                account_properties[sender_id]['account_holder'] = str(row['Sender_Account_Holder'])
            if pd.notna(row.get('Sender_Phone')):
                account_properties[sender_id]['phone'] = str(row['Sender_Phone'])
            if pd.notna(row.get('Sender_IP_Address')):
                account_properties[sender_id]['ip_address'] = str(row['Sender_IP_Address'])
            if pd.notna(row.get('Sender_City')):
                account_properties[sender_id]['city'] = str(row['Sender_City'])
            # Ensure is_fraud is boolean
            if pd.notna(row.get('is_fraud')):
                account_properties[sender_id]['is_fraud'] = bool(row['is_fraud'])


            # Update receiver properties based on transaction data
            current_receiver_fraud_flag = account_properties[receiver_id].get('Predicted_Fraud_Flag', 0)
            current_receiver_risk_score = account_properties[receiver_id].get('Predicted_Risk_Score', 0.0)

            if pd.notna(row.get('Predicted_Fraud_Flag')):
                new_fraud_flag = int(row['Predicted_Fraud_Flag'])
                if new_fraud_flag > current_receiver_fraud_flag: # Take the higher fraud flag
                    account_properties[receiver_id]['Predicted_Fraud_Flag'] = new_fraud_flag
            
            if pd.notna(row.get('Predicted_Risk_Score')):
                new_risk_score = float(row['Predicted_Risk_Score'])
                if new_risk_score > current_receiver_risk_score: # Take the higher risk score
                    account_properties[receiver_id]['Predicted_Risk_Score'] = new_risk_score

            # Add other account-specific properties if they exist in your Excel for receiver
            if pd.notna(row.get('Receiver_Account_Holder')):
                account_properties[receiver_id]['account_holder'] = str(row['Receiver_Account_Holder'])
            if pd.notna(row.get('Receiver_Phone')):
                account_properties[receiver_id]['phone'] = str(row['Receiver_Phone'])
            if pd.notna(row.get('Receiver_IP_Address')):
                account_properties[receiver_id]['ip_address'] = str(row['Receiver_IP_Address'])
            if pd.notna(row.get('Receiver_City')):
                account_properties[receiver_id]['city'] = str(row['Receiver_City'])
            # Ensure is_fraud is boolean
            if pd.notna(row.get('is_fraud')):
                account_properties[receiver_id]['is_fraud'] = bool(row['is_fraud'])


        # Add/Update Account nodes with all collected properties
        logging.info("Updating Account nodes with aggregated properties...")
        for account_id, props in account_properties.items():
            # Convert pandas NaN to None for Neo4j compatibility for all properties
            clean_props = {k: (None if pd.isna(v) else v) for k, v in props.items()}
            
            # Ensure 'is_fraud' is explicitly boolean, default to False if not present
            clean_props['is_fraud'] = bool(clean_props.get('is_fraud', False))
            
            # Ensure 'Predicted_Fraud_Flag' is integer, default to 0
            clean_props['Predicted_Fraud_Flag'] = int(clean_props.get('Predicted_Fraud_Flag', 0))
            
            # Ensure 'Predicted_Risk_Score' is float, default to 0.0
            clean_props['Predicted_Risk_Score'] = float(clean_props.get('Predicted_Risk_Score', 0.0))

            # Build SET clauses for ON MATCH
            # Exclude 'account_id' from the SET clause itself, as it's used in MERGE condition
            set_clauses_on_match = ", ".join([f"a.{key} = ${key}" for key in clean_props.keys() if key != 'account_id'])
            
            # Parameters for the query. This dictionary will contain all properties for both ON CREATE and ON MATCH.
            query_params = {"account_id": account_id, **clean_props}

            if set_clauses_on_match:
                # Use $props for ON CREATE and individual parameters for ON MATCH
                query = f"""
                    MERGE (a:Account {{account_id: $account_id}})
                    ON CREATE SET a = $props
                    ON MATCH SET {set_clauses_on_match}
                """
                # Add the 'props' key for ON CREATE
                query_params['props'] = clean_props
            else:
                # If no other properties to set, just MERGE
                query = f"""
                    MERGE (a:Account {{account_id: $account_id}})
                """
            
            session.run(query, query_params)

        logging.info(f"Created/Updated {len(account_properties)} unique Account nodes with enhanced properties.")


        # Iterate through DataFrame to create relationships
        logging.info("Starting data import into Neo4j (relationships)...")
        for index, row in df.iterrows():
            sender_account_id = str(row['Sender_Account'])
            receiver_account_id = str(row['Receiver_Account'])

            # Create TRANSACTION Relationship with all features as properties
            # Convert NaN values to None for Neo4j compatibility
            transaction_props = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            
            # Ensure numeric types are correctly cast for transaction properties as well
            if 'Predicted_Fraud_Flag' in transaction_props and transaction_props['Predicted_Fraud_Flag'] is not None:
                transaction_props['Predicted_Fraud_Flag'] = int(transaction_props['Predicted_Fraud_Flag'])
            if 'Predicted_Risk_Score' in transaction_props and transaction_props['Predicted_Risk_Score'] is not None:
                transaction_props['Predicted_Risk_Score'] = float(transaction_props['Predicted_Risk_Score'])
            if 'Amount' in transaction_props and transaction_props['Amount'] is not None:
                transaction_props['Amount'] = float(transaction_props['Amount'])

            # Determine if this is a 'Mule' transaction
            is_mule_transaction = (str(row.get('Sender_Bank', '')).lower() == 'mule' or str(row.get('Receiver_Bank', '')).lower() == 'mule')
            transaction_props['is_mule_transaction'] = is_mule_transaction # Add new property

            session.run("""
                MATCH (s:Account {account_id: $sender_id})
                MATCH (r:Account {account_id: $receiver_id})
                CREATE (s)-[t:TRANSACTION]->(r)
                SET t = $props
            """, sender_id=sender_account_id, receiver_id=receiver_account_id, props=transaction_props)

            if (index + 1) % 1000 == 0:
                logging.info(f"Processed {index + 1} transactions...")

        logging.info(f"Data import complete. Total nodes: {len(account_properties)}, Total relationships: {len(df)}")

    logging.info("\n--- Data Loading Complete ---")
    logging.info("Now, you can start your Flask backend (app.py) and interact with the data.")

except Exception as e:
    logging.error(f"An error occurred during Neo4j operation: {e}")
    logging.error(traceback.format_exc())
finally:
    if driver:
        driver.close()
        logging.info("Neo4j driver closed.")
