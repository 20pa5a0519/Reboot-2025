# app.py
from flask import Flask, request, jsonify
from neo4j import GraphDatabase, basic_auth
from flask_cors import CORS
import logging
from neo4j.time import DateTime, Date # Import Date type as well
import pandas as pd
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access

# --- Neo4j Connection Details ---
# IMPORTANT: Ensure your Neo4j database is running and accessible at this URI
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "Monmay#1699" # Replace with your actual Neo4j password

# Initialize Neo4j driver
driver = None
try:
    driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
    driver.verify_connectivity()
    logging.info("Neo4j driver initialized and connected successfully!")
except Exception as e:
    logging.error(f"Failed to connect to Neo4j: {e}")
    driver = None # Ensure driver is None if connection fails

@app.route('/')
def index():
    return "Neo4j Graph API is running. Use /api/account/<account_id> to query."

# Helper function to process properties, converting DateTime and Date objects
def process_properties(props):
    processed_props = {}
    for key, value in props.items():
        if isinstance(value, DateTime) or isinstance(value, Date): # Handle both DateTime and Date
            processed_props[key] = value.isoformat() # Convert to ISO string
        else:
            processed_props[key] = value
    return processed_props

@app.route('/api/account/<account_id>', methods=['GET'])
def get_account_graph(account_id):
    """
    Fetches graph data for a specific account_id, including direct transactions
    and a 'COLLUSION_PATTERN' based on multiple transactions with same timeframe,
    location, amount, IP, and browser.
    The account_id parameter can now be an account_id or a phone number.
    """
    if not driver:
        logging.error("Attempted to query Neo4j, but driver is not connected.")
        return jsonify({"error": "Neo4j database not connected."}), 500

    nodes = []
    edges = []
    node_ids = set() # To keep track of unique nodes already added (using account_id or unique ID)
    edge_keys = set() # To keep track of unique edges (using a combination of from-to-type)

    try:
        with driver.session() as session:
            # Cypher query to get the central node and its direct TRANSACTION connections.
            # Only include the COLLUSION_PATTERN logic.
            query = """
            MATCH (a:Account)
            WHERE a.account_id = $account_id OR a.phone = $account_id

            // Direct Transactions
            OPTIONAL MATCH (a)-[t:TRANSACTION]-(b:Account)

            // Find potential collusion patterns for transactions involving 'a' and 'collusion_b'
            // This pattern checks for multiple transactions between the same two accounts
            // within a close timeframe, same location, similar amounts, same IP, and same Browser.
            OPTIONAL MATCH (a)-[t1:TRANSACTION]-(collusion_b:Account)
            OPTIONAL MATCH (a)-[t2:TRANSACTION]-(collusion_b:Account)
            WHERE elementId(t1) <> elementId(t2) // Ensure distinct transactions
              AND t1.Geo_Location IS NOT NULL AND t1.Geo_Location = t2.Geo_Location
              AND t1.Timestamp IS NOT NULL AND t2.Timestamp IS NOT NULL
              AND abs(duration.inSeconds(t1.Timestamp, t2.Timestamp).seconds) <= 3600 // Within 1 hour
              AND t1.Amount IS NOT NULL AND t2.Amount IS NOT NULL
              AND abs(t1.Amount - t2.Amount) <= 0.1 * t1.Amount // Amounts within 10% difference
              AND t1.IP_Address IS NOT NULL AND t1.IP_Address = t2.IP_Address // New: Same IP Address
              AND t1.Browser IS NOT NULL AND t1.Browser = t2.Browser // New: Same Browser
            WITH a, t, b,
                 collect(DISTINCT {
                     from_id: a.account_id,
                     to_id: collusion_b.account_id,
                     type: 'COLLUSION_PATTERN',
                     color: 'red',
                     properties: {
                         pattern_description: 'Multiple transactions with similar time, location, amount, IP, and browser.',
                         location: t1.Geo_Location,
                         time_diff_seconds: abs(duration.inSeconds(t1.Timestamp, t2.Timestamp).seconds),
                         amount_diff_percentage: toFloat(abs(t1.Amount - t2.Amount)) / t1.Amount * 100,
                         ip_address: t1.IP_Address, // Add IP to properties
                         browser: t1.Browser, // Add Browser to properties
                         fraud_flag_t1: t1.Fraud_Flag, // Add fraud details from contributing transactions
                         risk_score_t1: t1.Risk_Score,
                         fraud_flag_t2: t2.Fraud_Flag,
                         risk_score_t2: t2.Risk_Score
                     }
                 }) AS collusion_patterns_data

            RETURN
                a,
                CASE
                    WHEN a.account_id = $account_id THEN 'gold' // Highlight searched account if it's the account_id
                    WHEN a.phone = $account_id THEN 'gold' // Highlight searched account if it's the phone number
                    WHEN a.is_fraud = true THEN 'red'
                    ELSE 'green'
                END AS a_color,
                t,
                // Calculate t_color based on Fraud_Flag and Risk_Score on the transaction relationship
                CASE
                    WHEN t.Fraud_Flag = 1 AND t.Risk_Score > 3 THEN 'red'
                    ELSE 'green'
                END AS t_color,
                b,
                CASE WHEN b.is_fraud = true THEN 'red' ELSE 'green'
                END AS b_color,
                collusion_patterns_data
            LIMIT 200 // Limit the number of results for performance in UI
            """
            result = session.run(query, account_id=account_id)

            for record in result:
                node_a = record['a']
                node_b = record['b']
                rel_t = record['t']
                collusion_patterns_data = record['collusion_patterns_data']

                # Helper to add a node to the list if not already present and build its tooltip title
                def add_node_if_new(node_obj, color_override=None):
                    # Determine node ID robustly: prefer 'account_id', then 'id', then 'element_id' if it's a Neo4j object
                    node_id_val = None
                    if isinstance(node_obj, dict):
                        node_id_val = node_obj.get('account_id') or node_obj.get('id')
                    elif hasattr(node_obj, 'element_id'): # It's a Neo4j object
                        node_id_val = node_obj.get('account_id') or str(node_obj.element_id)
                    
                    if node_id_val and node_id_val not in node_ids:
                        node_data = process_properties(node_obj)
                        node_data['id'] = node_id_val
                        node_data['label'] = node_obj.get('account_id', str(node_id_val)) # Prioritize account_id for label
                        node_data['color'] = color_override if color_override else node_data.get('color', 'green') # Default to green if no color

                        # Construct the HTML string for the tooltip (title property)
                        tooltip_html = f"<strong>ID:</strong> {node_data['id']}"
                        if node_data.get('label') and node_data['label'] != node_data['id']: tooltip_html += f"<br><strong>Label:</strong> {node_data['label']}"
                        if node_data.get('account_holder'): tooltip_html += f"<br><strong>Holder:</strong> {node_data['account_holder']}"
                        if node_data.get('phone'): tooltip_html += f"<br><strong>Phone:</strong> {node_data['phone']}"
                        
                        # Add detailed fraud info for nodes
                        if node_data.get('is_fraud') is not None:
                            tooltip_html += f"<br><strong>Fraudulent:</strong> {'Yes' if node_data['is_fraud'] else 'No'}"
                            if node_data['is_fraud']:
                                if node_data.get('Fraud_Flag') is not None: tooltip_html += f"<br><strong>Fraud Flag:</strong> {node_data['Fraud_Flag']}"
                                if node_data.get('Risk_Score') is not None: tooltip_html += f"<br><strong>Risk Score:</strong> {node_data['Risk_Score']}"
                                if node_data.get('fraud_reason'): tooltip_html += f"<br><strong>Reason:</strong> {node_data['fraud_reason']}"
                                if node_data.get('fraud_type'): tooltip_html += f"<br><strong>Type:</strong> {node_data['fraud_type']}"

                        if node_data.get('ip_address'): tooltip_html += f"<br><strong>IP:</strong> {node_data['ip_address']}"
                        if node_data.get('city'): tooltip_html += f"<br><strong>City:</strong> {node_data['city']}"

                        # Add any other dynamic properties to tooltip
                        for key, value in node_data.items():
                            if key not in ['id', 'label', 'color', 'account_holder', 'phone', 'is_fraud',
                                           'ip_address', 'city', 'title', 'Fraud_Flag', 'Risk_Score',
                                           'fraud_reason', 'fraud_type']: # Exclude 'title' itself
                                tooltip_html += f"<br><strong>{key.replace('_', ' ').title()}:</strong> {value}"

                        node_data['title'] = tooltip_html # Set the title property for Vis.js tooltip

                        nodes.append(node_data)
                        node_ids.add(node_id_val)

                # Helper to add an edge to the list if not already present and build its tooltip title
                def add_edge_if_new(from_obj, to_obj, label, color, properties=None):
                    # Determine from/to IDs robustly
                    from_id = None
                    if isinstance(from_obj, dict):
                        from_id = from_obj.get('account_id') or from_obj.get('id')
                    elif hasattr(from_obj, 'element_id'):
                        from_id = from_obj.get('account_id') or str(from_obj.element_id)
                    else: # Assume it's already the ID string
                        from_id = from_obj

                    to_id = None
                    if isinstance(to_obj, dict):
                        to_id = to_obj.get('account_id') or to_obj.get('id')
                    elif hasattr(to_obj, 'element_id'):
                        to_id = to_obj.get('account_id') or str(to_obj.element_id)
                    else: # Assume it's already the ID string
                        to_id = to_obj

                    edge_key = f"{from_id}-{to_id}-{label}"
                    if edge_key not in edge_keys:
                        edge_data = {
                            'from': from_id,
                            'to': to_id,
                            'label': label,
                            'color': color,
                            'arrows': 'to'
                        }
                        if properties:
                            edge_data.update(process_properties(properties))
                        
                        # Construct the HTML string for the tooltip (title property)
                        tooltip_html = f"<strong>Type:</strong> {label}"
                        if edge_data.get('Amount'): tooltip_html += f"<br><strong>Amount:</strong> {edge_data['Amount']} {edge_data.get('Currency', '')}"
                        if edge_data.get('Timestamp'): tooltip_html += f"<br><strong>Date:</strong> {pd.to_datetime(edge_data['Timestamp']).strftime('%Y-%m-%d %H:%M:%S')}" # Format timestamp
                        if edge_data.get('is_mule_transaction') is not None: tooltip_html += f"<br><strong>Mule Transaction:</strong> {'Yes' if edge_data['is_mule_transaction'] else 'No'}"
                        
                        # Add detailed fraud info for TRANSACTION edges
                        if label == 'TRANSACTION':
                            if edge_data.get('Fraud_Flag') is not None: tooltip_html += f"<br><strong>Fraud Flag:</strong> {edge_data['Fraud_Flag']}"
                            if edge_data.get('Risk_Score') is not None: tooltip_html += f"<br><strong>Risk Score:</strong> {edge_data['Risk_Score']}"
                            if edge_data.get('IP_Address'): tooltip_html += f"<br><strong>IP:</strong> {edge_data['IP_Address']}"
                            if edge_data.get('Geo_Location'): tooltip_html += f"<br><strong>Location:</strong> {edge_data['Geo_Location']}"
                            if edge_data.get('Browser'): tooltip_html += f"<br><strong>Browser:</strong> {edge_data['Browser']}"
                            if edge_data.get('Device_Type'): tooltip_html += f"<br><strong>Device Type:</strong> {edge_data['Device_Type']}"
                            if edge_data.get('KYC_Fingerprinting'): tooltip_html += f"<br><strong>KYC:</strong> {edge_data['KYC_Fingerprinting']}"
                        elif label == 'COLLUSION_PATTERN':
                            tooltip_html += f"<br><br><strong>Collusion Details:</strong>"
                            if edge_data.get('pattern_description'): tooltip_html += f"<br><strong>Pattern:</strong> {edge_data['pattern_description']}"
                            if edge_data.get('location'): tooltip_html += f"<br><strong>Common Location:</strong> {edge_data['location']}"
                            if edge_data.get('ip_address'): tooltip_html += f"<br><strong>Common IP:</strong> {edge_data['ip_address']}" # New: Common IP
                            if edge_data.get('browser'): tooltip_html += f"<br><strong>Common Browser:</strong> {edge_data['browser']}" # New: Common Browser
                            if edge_data.get('time_diff_seconds') is not None: tooltip_html += f"<br><strong>Time Difference:</strong> {edge_data['time_diff_seconds']} seconds"
                            if edge_data.get('amount_diff_percentage') is not None: tooltip_html += f"<br><strong>Amount Difference:</strong> {edge_data['amount_diff_percentage']:.2f}%"
                            # Add fraud flags/risk scores from contributing transactions if available
                            if edge_data.get('fraud_flag_t1') is not None: tooltip_html += f"<br><strong>Trans 1 Fraud Flag:</strong> {edge_data['fraud_flag_t1']}"
                            if edge_data.get('risk_score_t1') is not None: tooltip_html += f"<br><strong>Trans 1 Risk Score:</strong> {edge_data['risk_score_t1']}"
                            if edge_data.get('fraud_flag_t2') is not None: tooltip_html += f"<br><strong>Trans 2 Fraud Flag:</strong> {edge_data['fraud_flag_t2']}"
                            if edge_data.get('risk_score_t2') is not None: tooltip_html += f"<br><strong>Trans 2 Risk Score:</strong> {edge_data['risk_score_t2']}"


                        # Add any other dynamic properties to tooltip
                        for key, value in edge_data.items():
                            # Exclude 'title' itself and already handled keys
                            if key not in ['from', 'to', 'label', 'color', 'arrows', 'Amount', 'Currency', 'Timestamp',
                                           'is_mule_transaction', 'Fraud_Flag', 'Risk_Score', 'IP_Address', 'Geo_Location',
                                           'Browser', 'Device_Type', 'KYC_Fingerprinting', 'title', 'ip_address', 'city',
                                           'pattern_description', 'location', 'time_diff_seconds', 'amount_diff_percentage',
                                           'fraud_flag_t1', 'risk_score_t1', 'fraud_flag_t2', 'risk_score_t2', 'browser']:
                                if value is not None: # Only add if value is not None
                                    tooltip_html += f"<br><strong>{key.replace('_', ' ').title()}:</strong> {value}"

                        edge_data['title'] = tooltip_html # Set the title property for Vis.js tooltip

                        edges.append(edge_data)
                        edge_keys.add(edge_key)

                # Process central node 'a'
                add_node_if_new(node_a, record['a_color'])

                # Process connected node 'b' from TRANSACTION
                if node_b:
                    add_node_if_new(node_b, record['b_color'])

                    if rel_t:
                        add_edge_if_new(
                            rel_t.start_node,
                            rel_t.end_node,
                            rel_t.type,
                            record['t_color'], # Use the calculated t_color
                            rel_t
                        )
                
                # Process COLLUSION_PATTERN edges
                if collusion_patterns_data:
                    for pattern_edge in collusion_patterns_data:
                        # Ensure the 'to' node for the pattern exists in our nodes list
                        # The 'from' node is 'a' which is already added
                        add_node_if_new(
                            {'account_id': pattern_edge['to_id']}, # Create a dummy node object for add_node_if_new
                            'red' # Collusion pattern nodes should also be red if involved
                        )
                        add_edge_if_new(
                            {'account_id': pattern_edge['from_id']}, # Dummy node object for from
                            {'account_id': pattern_edge['to_id']},   # Dummy node object for to
                            pattern_edge['type'],
                            pattern_edge['color'],
                            pattern_edge['properties']
                        )


            # If no nodes are found for the given account_id, return a 404
            if not nodes:
                logging.info(f"No graph data found for account_id: {account_id}")
                return jsonify({"nodes": [], "edges": []}), 404 # Return 404 if no data found

            return jsonify({"nodes": nodes, "edges": edges})

    except Exception as e:
        logging.error(f"Error fetching graph data for {account_id}: {e}")
        return jsonify({"error": f"Error fetching graph data: {e}"}), 500

@app.route('/api/account/update/<account_id>', methods=['PUT'])
def update_account_properties(account_id):
    """
    Updates properties of an existing Account node in Neo4j.
    Expects a JSON body with key-value pairs of properties to update.
    Example: {"is_fraud": true, "city": "New York"}
    """
    if not driver:
        logging.error("Attempted to update Neo4j, but driver is not connected.")
        return jsonify({"error": "Neo4j database not connected."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain a JSON body with properties to update."}), 400

        # Construct SET clause dynamically
        set_clauses = [f"n.{key} = ${key}" for key in data.keys()]
        set_query = ", ".join(set_clauses)

        query = f"""
        MATCH (n:Account {{account_id: $account_id}})
        SET {set_query}
        RETURN n
        """
        
        # Add account_id to the parameters for the MATCH clause
        params = {"account_id": account_id, **data}

        with driver.session() as session:
            result = session.run(query, params)
            updated_node = result.single()

            if updated_node:
                logging.info(f"Account {account_id} updated successfully with properties: {data}")
                return jsonify({"message": f"Account {account_id} updated successfully.", "updated_properties": data}), 200
            else:
                logging.warning(f"Account {account_id} not found for update.")
                return jsonify({"error": f"Account {account_id} not found."}), 404

    except Exception as e:
        logging.error(f"Error updating account {account_id}: {e}")
        return jsonify({"error": f"Error updating account: {e}"}), 500

@app.route('/api/upload_excel_update', methods=['POST'])
def upload_excel_update():
    """
    Receives an Excel file, reads it, and updates/creates Account nodes in Neo4j.
    Expected Excel columns: account_id (mandatory), and any other properties.
    """
    if not driver:
        logging.error("Attempted to update Neo4j via Excel, but driver is not connected.")
        return jsonify({"error": "Neo4j database not connected."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({"error": "Invalid file type. Please upload an Excel file (.xlsx or .xls)."}), 400

    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(io.BytesIO(file.read()))

        if 'account_id' not in df.columns:
            return jsonify({"error": "Excel file must contain an 'account_id' column."}), 400

        updated_count = 0
        created_count = 0
        errors = []

        with driver.session() as session:
            for index, row in df.iterrows():
                try:
                    account_id = str(row['account_id'])
                    properties_to_set = {}
                    for col_name, value in row.items():
                        if col_name != 'account_id':
                            # Convert pandas NaN to None for Neo4j
                            if pd.isna(value):
                                properties_to_set[col_name] = None
                            else:
                                properties_to_set[col_name] = value

                    # Cypher query to MERGE (create or match) the Account node
                    # and SET its properties.
                    # MERGE ensures idempotency: if it exists, match; if not, create.
                    set_clauses = ", ".join([f"n.{key} = ${key}" for key in properties_to_set.keys()])
                    
                    # Only apply SET if there are properties to set, otherwise just MERGE
                    if set_clauses:
                        query = f"""
                        MERGE (n:Account {{account_id: $account_id}})
                        ON CREATE SET n = $properties
                        ON MATCH SET {set_clauses}
                        RETURN n, labels(n) as labels
                        """
                        params = {"account_id": account_id, "properties": properties_to_set, **properties_to_set}
                    else:
                        query = f"""
                        MERGE (n:Account {{account_id: $account_id}})
                        RETURN n, labels(n) as labels
                        """
                        params = {"account_id": account_id}

                    result = session.run(query, params)
                    record = result.single()
                    if record:
                        updated_count += 1
                except Exception as e:
                    errors.append(f"Error processing account_id {account_id}: {e}")
                    logging.error(f"Error processing account_id {account_id}: {e}")

        if errors:
            return jsonify({
                "message": f"Processed {df.shape[0]} rows. Updated/Created: {updated_count}. Errors: {len(errors)}.",
                "errors": errors
            }), 200 # Return 200 even with errors, but include error details
        else:
            return jsonify({
                "message": f"Successfully processed {df.shape[0]} rows. All accounts updated/created.",
                "updated_count": updated_count
            }), 200

    except Exception as e:
        logging.error(f"Error processing Excel file: {e}")
        return jsonify({"error": f"Failed to process Excel file: {e}"}), 500

if __name__ == '__main__':
    if driver:
        app.run(debug=True, port=5000)
    else:
        logging.error("Flask app cannot start without a successful Neo4j connection. Please check your Neo4j server and credentials.")
