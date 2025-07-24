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
password = "sonu@Mule3" # Default password, change if you've set a new one

# --- 2. Load Data from Excel ---
excel_file_path = 'D:\MEAN ACC\Reboot_25\database\hope_final.xlsx'
sheet_name = 'hope_final'

# Initialize variables outside the try block to ensure they are always defined
df = pd.DataFrame() # Initialize an empty DataFrame
unique_accounts = set() # Initialize an empty set
all_fraudulent_accounts = set() # Initialize an empty set

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

        # Determine fraudulent accounts
        fraudulent_senders = set(df[df['Predicted_Fraud_Flag'] == 1]['Sender_Account'])
        fraudulent_receivers = set(df[df['Predicted_Fraud_Flag'] == 1]['Receiver_Account'])
        all_fraudulent_accounts = fraudulent_senders.union(fraudulent_receivers)

        # Add nodes (accounts) - This is where unique_accounts is defined
        unique_accounts = pd.concat([df['Sender_Account'], df['Receiver_Account']]).unique()
        for account_id in unique_accounts:
            is_fraud = account_id in all_fraudulent_accounts
            session.run("""
                MERGE (a:Account {account_id: $account_id})
                SET a.is_fraud = $is_fraud
            """, account_id=account_id, is_fraud=is_fraud)
        logging.info(f"Created {len(unique_accounts)} unique Account nodes.")


        # Iterate through DataFrame to create relationships
        logging.info("Starting data import into Neo4j (relationships)...")
        for index, row in df.iterrows():
            sender_account_id = row['Sender_Account']
            receiver_account_id = row['Receiver_Account']

            # Create TRANSACTION Relationship with all features as properties
            # Convert NaN values to None for Neo4j compatibility
            transaction_props = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            
            # Determine if this is a 'Mule' transaction
            is_mule_transaction = (row.get('Sender_Bank') == 'Mule' or row.get('Receiver_Bank') == 'Mule')
            transaction_props['is_mule_transaction'] = is_mule_transaction # Add new property

            session.run("""
                MATCH (s:Account {account_id: $sender_id})
                MATCH (r:Account {account_id: $receiver_id})
                CREATE (s)-[t:TRANSACTION]->(r)
                SET t = $props
            """, sender_id=sender_account_id, receiver_id=receiver_account_id, props=transaction_props)

            if (index + 1) % 1000 == 0:
                logging.info(f"Processed {index + 1} transactions...")

        logging.info(f"Data import complete. Total nodes: {len(unique_accounts)}, Total relationships: {len(df)}")

    logging.info("\n--- Data Loading Complete ---")
    logging.info("Now, you can start your Flask backend (app.py) and interact with the data.")

except Exception as e:
    logging.error(f"An error occurred during Neo4j operation: {e}")
    logging.error(traceback.format_exc())
finally:
    if driver:
        driver.close()
        logging.info("Neo4j driver closed.")
