const neo4j = require('neo4j-driver');

const driver = neo4j.driver(
  'bolt://127.0.0.1:7687',
  neo4j.auth.basic('neo4j', 'sonu@Mule3')
);

driver.verifyConnectivity()
  .then(() => console.log('✅ Connected to Neo4j'))
  .catch(err => console.error('❌ Failed to connect:', err))
  .finally(() => driver.close());
