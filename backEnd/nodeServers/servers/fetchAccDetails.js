let express = require("express");
let cors = require("cors");
let mongodb = require("mongodb");
const { Result } = require("neo4j-driver");

let app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(cors());

let mongoclient = mongodb.MongoClient;

let url = "mongodb://127.0.0.1:27017";
let dbName = "reboot";

let AId;

let storedResult = null;

let flag = null;

app.get("/transactions", (req, res) => {
  console.log("Connecting to MongoDB...");
  mongoclient
    .connect(url)
    .then((conn) => {
      console.log("connected successfully");
      const db = conn.db(dbName);
      let transactions = db
        .collection("transactions2")
        .find()
        .toArray()
        .then((data) => {
          console.log("all data fetched");

          transactions = Object.values(data);

          console.log(data.length);

          const senderMap = new Map();

          // Group transactions by Sender_Account
          transactions.forEach((txn) => {
            const sender = txn.Sender_Account;
            if (!senderMap.has(sender)) {
              senderMap.set(sender, []);
            }
            senderMap.get(sender).push(txn);
          });

          const muleAccounts = [];
          const legitAccounts = [];

          for (const [sender, txns] of senderMap.entries()) {
            const isMule = txns.some((t) => t.Predicted_Fraud_Flag === 1);
            const senderName = txns[0].Sender_Name || "Unknown";

            const accountObj = {
              id: sender,
              HolderName: senderName,
              status: isMule ? "Mule" : "Legit",
            };

            if (isMule) {
              muleAccounts.push(accountObj);
            } else {
              legitAccounts.push(accountObj);
            }
          }

          console.log(muleAccounts.length);
          console.log(legitAccounts.count);

          res.json({
            muleAccounts,
            legitAccounts,
            muleCount: muleAccounts.length,
            legitCount: legitAccounts.length,
          });
        })
        .catch((e) => {
          console.log("error fetching data", e);
          res.status(500).send("fetch failed");
          conn.close();
        });
    })
    .catch((err) => {
      console.log("error connecting", err);
      res.status(500).send("coonection failed");
    });
});

app.post("/detect", (req, res) => {
  console.log("entering first api");
  console.log(req.body);

  let AId = req.body.id;

  storedResult = null;

  mongoclient
    .connect(url)
    .then((conn) => {
      console.log("connected successfully to detect");
      const db = conn.db(dbName);
      return db
        .collection("transactions2")
        .find({ Sender_Account: AId })
        .toArray()
        .then((data) => {
          storedResult = data;
          console.log("storedResult");
          conn.close();
          res.json({ status: data ? "found" : "not found" });
        })
        .catch((e) => {
          console.log("error searching for data", e);
          conn.close();
        });
    })
    .catch((err) => {
      console.log("connection failed", err);
    });
});

app.get("/fetchResult", (req, res) => {
  console.log("entering 2nd api");
  if (storedResult) {
    res.json(storedResult);
    console.log("results sent");
  } else {
    res.status(404).json({ message: "No stored data found" });
  }
});

app.post("/live/:model", async (req, res) => {
  console.log(`Entering ${req.params.model} model check`);
  console.log(req.body);

  const {
    senderAccountId,
    receiverAccountId,
    amount,
    senderBank,
    receiverBank,
  } = req.body;

  mongoclient
    .connect(url)
    .then((conn) => {
      console.log("connected to VAE");
      const db = conn.db(dbName);
      const query = {
        Sender_Account: senderAccountId,
        Receiver_Account: receiverAccountId,
        Amount: amount,
        Sender_Bank: senderBank,
        Receiver_Bank: receiverBank,
      };
      return db
        .collection("transactions2")
        .find(query)
        .toArray()
        .then((data) => {
          if (data && data.length > 0) {
            txn = data[0]; // grab the first (or iterate over all)
            flag = txn.Predicted_Fraud_Flag;
            console.log(flag);
            res.json({ status: "found" });
          } else {
            res.json({ status: "not found" });
          }
        })
        .catch((e) => {
          console.log("error searching for data", e);
          conn.close();
        });
    })
    .catch((err) => {
      console.log("connection failed", err);
    });
});

app.get("/fetchflag", (req, res) => {
  console.log("entering VAE result api");
  console.log(flag);
  if (flag !== null) {
    res.json(flag);
    console.log("flag sent");
  } else {
    res.status(404).json({ message: "No stored data found" });
  }
});

let reasons = [];

// app.post("/searchReasons", (req, res) => {
//   console.log("inside search reasons");
//   console.log(req.body);

//   let AId = req.body.id;

//   mongoclient
//     .connect(url)
//     .then((conn) => {
//       console.log("connected to reasons db");
//       const db = conn.db(dbName);
//       return db
//         .collection("reasons")
//         .find({ Sender_Account: AId })
//         .toArray()
//         .then((data) => {
//           const reasonsArray = data.Fraud_Reason.split(";")
//             .map((r) => r.trim())
//             .filter((r) => r.length > 0)
//             .slice(0, 3); // get only the first 3 if more exist
//           console.log("reasons fetched");
//           console.log("Returning reasons:", reasonsArray);
//           reasons = reasonsArray;
//           conn.close();
//           return res.json({ status: data ? "found" : "not found" });
//         })
//         .catch((e) => {
//           console.log("error searching for reasons", e);
//           conn.close();
//         });
//     })
//     .catch((err) => {
//       console.error("Connection failed:", err);
//       res.status(500).json({ error: "MongoDB connection failed" });
//     });
// });

app.post("/searchReasons", (req, res) => {
  console.log("inside search reasons");
  console.log(req.body);

  const AId = req.body.id;

  mongoclient
    .connect(url)
    .then((conn) => {
      console.log("connected to reasons db");
      const db = conn.db(dbName);

      return db
        .collection("reasons")
        .find({ Sender_Account: AId })
        .toArray()
        .then((data) => {
          conn.close();

          if (!Array.isArray(data) || data.length === 0) {
            return res
              .status(404)
              .json({ error: "No reasons found for this account" });
          }

          // Grab the first matching document
          const doc = data[0];
          const fr = doc.Fraud_Reason;
          if (typeof fr !== "string") {
            return res
              .status(404)
              .json({ error: "Fraud_Reason field missing" });
          }

          const reasonsArray = fr
            .split(";")
            .map((r) => r.trim())
            .filter((r) => r.length > 0)
            .slice(0, 3); // up to 3 reasons

          console.log("reasons fetched");

          reasons = reasonsArray;

          return res.json({ reasons: reasonsArray });
        })
        .catch((e) => {
          console.error("error searching for reasons", e);
          conn.close();
          return res.status(500).json({ error: "Database query failed" });
        });
    })
    .catch((err) => {
      console.error("Connection failed:", err);
      return res.status(500).json({ error: "MongoDB connection failed" });
    });
});

app.get("/getReasons", (req, res) => {
  console.log("entering into get results");

  if (reasons) {
    res.json(reasons);
    console.log(reasons);
    console.log("reasons sent");
  } else {
    res.status(404).json({ message: "No reasons data found" });
  }
});

app.listen(7000, () => {
  console.log("listening on port 7000");
});
