# 🚚 Logistics Clearance Risk Prediction System

## From Descriptive Analytics to Predictive Decision Support

This project extends a logistics and supply chain analysis built in **Power BI** with a **Machine Learning and Streamlit prediction layer**.

The goal is to move logistics teams from simply understanding what happened to identifying **which shipments may require attention before a potential clearance issue becomes an operational problem.**

---

## 🎯 Business Problem

Traditional logistics dashboards provide valuable historical insights into:

- Shipment performance
- Carrier performance
- Trade flows
- Freight costs
- Customs clearance
- Customer and order performance

But reporting alone is largely retrospective.

This project asks:

> **Can historical shipment characteristics be used to predict customs clearance risk early enough to support better operational decisions?**

---

## 💡 Solution

The system uses two machine learning models:

### Regression
Predicts expected **customs clearance time in days**.

### Classification
Predicts the **probability of high clearance risk**.

The application then combines these outputs with shipment value to provide:

- Predicted clearance time
- High risk probability
- Risk classification
- Estimated value at risk
- Operational recommendation

---

## 💰 Why It Matters

A potential clearance delay can affect more than transportation time.

It can influence:

**Shipment availability → Inventory → Fulfillment → Customer Experience → Business Performance**

By identifying higher risk shipments earlier, organizations can potentially:

- Prioritize operational reviews
- Focus resources on higher risk shipments
- Improve exception management
- Review customs readiness earlier
- Understand potential shipment value exposure
- Support more proactive supply chain decisions

The model is designed as **decision support**, not a replacement for human expertise.

---

## 📊 Power BI + Machine Learning

This project combines descriptive and predictive analytics:

```text
Power BI
   ↓
What happened?
   ↓
Machine Learning
   ↓
What may happen?
   ↓
Streamlit
   ↓
What should we pay attention to?
