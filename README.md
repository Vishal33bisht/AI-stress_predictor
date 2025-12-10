# 🧠 AI-Powered Personal Schedule Optimization Assistant  
An end-to-end AI/ML system that analyzes your daily schedule, predicts stress levels, and automatically suggests optimized changes to rebalance your day.  
Built using **Python**, **scikit-learn**, **Streamlit**, and a custom **AI Auto-Rescheduler engine**.

---

## 🚀 Overview
This project is an intelligent productivity assistant that predicts how stressful your day will be, analyzes your time distribution, and uses AI-based reasoning to suggest a better version of your schedule.

The system combines:

- **Machine Learning** (RandomForest classifier)
- **Human-behavior modeling**
- **Scenario simulation**
- **AI optimization**
- **Dynamic UI (Streamlit)**
- **Multiple user profiles (Students, Office Workers, Others)**

This makes it a unique hybrid of **Data Science**, **Machine Learning**, and **AI-driven recommendation systems**.

---

## ✨ Key Features

### **1. ML-based Stress Prediction**
Predicts **Low / Medium / High** stress levels using:

- total busy hours  
- classes/meetings  
- deep work / study hours  
- commute  
- sleep  
- tasks due  
- context switching  
- engineered ratios (meeting load, deep-work ratio)

A trained **RandomForestClassifier** powers the inference.

---

### **2. What-If Scenario Analyzer**
Allows users to test alternate versions of their day:

- Sleep +1 hour  
- Meetings/classes –2  
- Deep-work +1 hour  
- Reduce context switching  
- Fewer deadline tasks  

Shows how each change would **affect stress level**.

---

### **3. 🤖 AI Auto-Rescheduler (Killer Feature)**
This is the standout component of the project.

The Auto-Rescheduler:

- Searches dozens of nearby day configurations  
- Uses the ML model to evaluate each possibility  
- Identifies the best plan that lowers stress  
- Produces actionable suggestions like:

> “Sleep 7 hours → Stress drops from HIGH to MEDIUM”  
> “Reduce assignments from 4 to 2”  
> “Lower busy hours from 10h to 8h”

This makes the system operate like a true **AI personal assistant**.

---

### **4. Profile-Based Personalization (Student / Office / Other)**
Different users have different patterns.

The app adjusts:

- Labels  
- Templates  
- Suggested behavior  
- Day presets  

Depending on whether the user is a:

- **Student**  
- **Office worker**  
- **Freelancer/Other**

---

### **5. Interactive Streamlit Dashboard**
The UI includes:

- Stress prediction card  
- Customized recommendations  
- Time-distribution bar charts  
- What-If table  
- Auto-rescheduler comparison (Current vs Proposed day)  
- Saved day history  
- Stress trend plot  
- CSV batch analysis  

Everything runs locally through Streamlit.

---

## 🏗️ Project Structure

