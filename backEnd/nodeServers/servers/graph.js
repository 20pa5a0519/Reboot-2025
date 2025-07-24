// const neo4j = require("neo4j-driver");
// const express = require("express");
// let cors = require("cors");

// const app = express();

// app.use(cors());

// const uri = "bolt://neo4j://127.0.0.1:7687"; // Replace with your actual DB ID
// const user = "neo4j";
// const password = "1Khr6LO-k0R6_hLfapzePLocqHUW-yTFtgVdD8MZ25A"; // Replace with your actual password

// const driver = neo4j.driver(uri, neo4j.auth.basic(user, password));

// app.get("/graph/:id", async (req, res) => {
//   const id = req.params.id;
//   const session = driver.session();

//   try {
//     const result = await session.run(
//       "MATCH (a:Account {id: $id})-[:CONNECTED_TO*1..3]-(b:Account) RETURN a, b",
//       { id }
//     );

//     const records = result.records.map((record) => ({
//       a: record.get("a").properties,
//       b: record.get("b").properties,
//     }));

//     res.json(records);
//     console.log(records);
//   } catch (error) {
//     console.error("Neo4j query error:", error);
//     res.status(500).send("Neo4j query failed");
//   } finally {
//     await session.close();
//   }
// });

// app.listen(3000, () => {
//   console.log("Server is running on http://localhost:3000");
// });

// backend.js (Node.js with Express and Neo4j)

// const express = require("express");
// const bodyParser = require("body-parser");
// const cors = require("cors");
// const neo4j = require("neo4j-driver");

// const app = express();
// const port = 3000;

// // Middleware
// app.use(cors());
// app.use(bodyParser.json());

// // Neo4j connection setup
// const driver = neo4j.driver(
//   "bolt://localhost:7687",
//   neo4j.auth.basic("neo4j", "sonu@Mule3")
// );

// driver
//   .verifyConnectivity()
//   .then(() => console.log("✅ Connected to Neo4j"))
//   .catch((err) => console.error("❌ Failed to connect:", err))
//   .finally(() => driver.close());

// // Endpoint: Check health
// app.get("/", (req, res) => {
//   res.send("Neo4j Graph API (Node.js) is running.");
// });

// // Fetch graph for given account ID or phone
// app.get("/graph/:accountId", async (req, res) => {
//   console.log(req);
//   const accountId = req.params.id;

//   const session = driver.session();
//   try {
//     const result = await session.run(
//       `
//       MATCH (a:Account)
//       WHERE a.account_id = $accountId OR a.phone = $accountId

//       OPTIONAL MATCH (a)-[t:TRANSACTION]-(b:Account)
//       OPTIONAL MATCH (a)-[t1:TRANSACTION]-(collusion_b:Account)
//       OPTIONAL MATCH (a)-[t2:TRANSACTION]-(collusion_b:Account)
//       WHERE elementId(t1) <> elementId(t2)
//         AND t1.Geo_Location IS NOT NULL AND t1.Geo_Location = t2.Geo_Location
//         AND t1.Timestamp IS NOT NULL AND t2.Timestamp IS NOT NULL
//         AND abs(duration.inSeconds(t1.Timestamp, t2.Timestamp).seconds) <= 3600
//         AND t1.Amount IS NOT NULL AND t2.Amount IS NOT NULL
//         AND abs(t1.Amount - t2.Amount) <= 0.1 * t1.Amount
//         AND t1.IP_Address = t2.IP_Address
//         AND t1.Browser = t2.Browser

//       WITH a, t, b,
//         collect(DISTINCT {
//           from_id: a.account_id,
//           to_id: collusion_b.account_id,
//           type: 'COLLUSION_PATTERN',
//           color: 'red',
//           properties: {
//             pattern_description: 'Multiple transactions with similar time, location, amount, IP, and browser.',
//             location: t1.Geo_Location,
//             time_diff_seconds: abs(duration.inSeconds(t1.Timestamp, t2.Timestamp).seconds),
//             amount_diff_percentage: toFloat(abs(t1.Amount - t2.Amount)) / t1.Amount * 100,
//             ip_address: t1.IP_Address,
//             browser: t1.Browser,
//             fraud_flag_t1: t1.Predicted_Fraud_Flag,
//             risk_score_t1: t1.Predicted_Risk_Score,
//             fraud_flag_t2: t2.Predicted_Fraud_Flag,
//             risk_score_t2: t2.Predicted_Risk_Score
//           }
//         }) AS collusion_patterns_data

//       RETURN a, t, b, collusion_patterns_data
//       LIMIT 200
//     `,
//       { accountId }
//     );

//     const nodes = [];
//     const edges = [];
//     const nodeIds = new Set();
//     const edgeKeys = new Set();

//     result.records.forEach((record) => {
//       const a = record.get("a").properties;
//       const b = record.get("b")?.properties;
//       const t = record.get("t")?.properties;
//       const collusionData = record.get("collusion_patterns_data");

//       const addNode = (node) => {
//         const id = node.account_id;
//         if (!nodeIds.has(id)) {
//           nodes.push({ id, label: id, ...node });
//           nodeIds.add(id);
//         }
//       };

//       const addEdge = (fromId, toId, label, color, props) => {
//         const key = `${fromId}-${toId}-${label}`;
//         if (!edgeKeys.has(key)) {
//           edges.push({ from: fromId, to: toId, label, color, ...props });
//           edgeKeys.add(key);
//         }
//       };

//       if (a) addNode(a);
//       if (b) addNode(b);
//       if (t && a && b)
//         addEdge(a.account_id, b.account_id, "TRANSACTION", "green", t);

//       if (collusionData) {
//         collusionData.forEach((cd) => {
//           const { from_id, to_id, type, color, properties } = cd;
//           addNode({ account_id: to_id });
//           addEdge(from_id, to_id, type, color, properties);
//         });
//       }
//     });

//     if (nodes.length === 0) {
//       return res.status(404).json({ nodes: [], edges: [] });
//     }

//     res.json({ nodes, edges });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ error: "Internal server error" });
//   } finally {
//     await session.close();
//   }
// });

// app.listen(port, () => {
//   console.log(`Server is running on port ${port}`);
// });

// backend.js
const express = require("express");
const cors = require("cors");
const neo4j = require("neo4j-driver");

const app = express();
app.use(express.json());

app.use(cors());

const driver = neo4j.driver(
  "bolt://localhost:7687",
  neo4j.auth.basic("neo4j", "sonu@Mule3")
);

app.get("/graph/:id", async (req, res) => {
  const accountId = req.params.id;
  const session = driver.session();

  try {
    const query = `
      MATCH (a:Account)
WHERE a.account_id = $accountId OR a.phone = $accountId

// Basic transaction links
OPTIONAL MATCH (a)-[t:TRANSACTION]-(b:Account)

// Collusion pattern detection
OPTIONAL MATCH (a)-[t1:TRANSACTION]-(collusion_b:Account),
              (a)-[t2:TRANSACTION]-(collusion_b)
WHERE elementId(t1) <> elementId(t2)
  AND t1.Geo_Location IS NOT NULL AND t1.Geo_Location = t2.Geo_Location
  AND t1.Timestamp IS NOT NULL AND t2.Timestamp IS NOT NULL
  AND abs(duration.inSeconds(t1.Timestamp, t2.Timestamp).seconds) <= 3600
  AND t1.Amount IS NOT NULL AND t2.Amount IS NOT NULL
  AND t1.Amount <> 0
  AND abs(t1.Amount - t2.Amount) <= 0.1 * t1.Amount
  AND t1.IP_Address = t2.IP_Address
  AND t1.Browser = t2.Browser

WITH a, b, collusion_b, t1, t2
WITH a, collect(DISTINCT { id: b.account_id }) AS basic_links,
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
         ip_address: t1.IP_Address,
         browser: t1.Browser,
         fraud_flag_t1: t1.Predicted_Fraud_Flag,
         risk_score_t1: t1.Predicted_Risk_Score,
         fraud_flag_t2: t2.Predicted_Fraud_Flag,
         risk_score_t2: t2.Predicted_Risk_Score
       }
     }) AS collusion_patterns_data

RETURN a, basic_links, collusion_patterns_data
LIMIT 200

    `;

    const result = await session.run(query, { accountId });

    let nodesMap = {};
    let links = [];
    let details = [];

    result.records.forEach((row) => {
      const aNode = row.get("a").properties;
      if (!aNode) return;

      nodesMap[aNode.account_id] = {
        id: aNode.account_id,
        label: aNode.name || aNode.account_id,
      };

      // row.get('basic_links').forEach(b => {
      //   nodesMap[b.id] = { id: b.id, label: b.id };
      //   links.push({ source: aNode.account_id, target: b.id, type: 'TRANSACTION', color: 'steelblue' });
      // });

      // row.get('collusion_patterns_data').forEach(cp => {
      //   const { from_id, to_id, type, color, properties } = cp;
      //   links.push({ source: from_id, target: to_id, type, color, properties });

      // });

      const basicLinks = row.get("basic_links");
      if (Array.isArray(basicLinks)) {
        basicLinks.forEach((b) => {
          if (!nodesMap[b.id]) {
            nodesMap[b.id] = { id: b.id, label: b.id };
          }
          links.push({
            source: aNode.account_id,
            target: b.id,
            type: "TRANSACTION",
            color: "steelblue",
          });
        });
      }

      const collusionPatterns = row.get("collusion_patterns_data");
      console.log("Raw row data:", row);
      console.log(
        "collusion_patterns_data:",
        row.get("collusion_patterns_data")
      );
      if (Array.isArray(collusionPatterns)) {
        collusionPatterns.forEach((cp) => {
          if (cp?.from_id && cp?.to_id) {
            // Add nodes if not already present
            if (!nodesMap[cp.from_id]) {
              nodesMap[cp.from_id] = { id: cp.from_id, label: cp.from_id };
            }
            if (!nodesMap[cp.to_id]) {
              nodesMap[cp.to_id] = { id: cp.to_id, label: cp.to_id };
            }

            links.push({
              source: cp.from_id,
              target: cp.to_id,
              type: cp.type,
              color: cp.color,
              properties: cp.properties || {},
            });

            details.push({
              from: cp.from_id,
              to: cp.to_id,
              description: cp.properties?.pattern_description,
              time_diff: cp.properties?.time_diff_seconds,
              amount_diff: cp.properties?.amount_diff_percentage,
              location: cp.properties?.location,
              ip: cp.properties?.ip_address,
              browser: cp.properties?.browser,
              fraud_score_1: cp.properties?.risk_score_t1,
              fraud_score_2: cp.properties?.risk_score_t2,
            });
          }
        });
      }
    });

    const nodes = Object.values(nodesMap);
    console.log(nodes, links, details);
    res.json({ nodes, links, details });
  } catch (error) {
    console.error("Graph API error:", error);
    res.status(500).json({ message: "Error fetching graph data" });
  } finally {
    await session.close();
  }
});

app.listen(3000, () => console.log("Graph API running on port 3000"));
