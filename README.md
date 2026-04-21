# Conversational BI Agent — Instacart Dataset
 
A natural language Business Intelligence agent built on the Instacart Market Basket Analysis dataset (~3.4M orders, 206K users, 50K products across 6 interrelated CSV files). Ask questions in plain English and get back SQL queries, interactive charts, data tables, and AI-generated insights.
 
---
 
## Demo
 
```
You:   "Which departments have the highest reorder rate?"
Agent: Runs a 3-table JOIN → bar chart → "The produce department leads with a 67%
        reorder rate, suggesting customers reliably return for fresh items weekly."
```
 
---

 
## Setup
 
### Prerequisites
 
- Python 3.10+
- Node.js 18+ (for frontend)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
 
### 1. Clone and install backend dependencies
 
```bash
cd backend
pip install -r requirements.txt
```
 
### 2. Add your CSVs
 
```
project-root/
├── backend/
├── frontend/
└── data/               ← place all 6 CSVs here
    ├── orders.csv
    ├── order_products__prior.csv
    ├── order_products__train.csv
    ├── products.csv
    ├── aisles.csv
    └── departments.csv
```
 
### 3. Set your Gemini API key
 
**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your_key_here
```
 
**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your_key_here"
```
 
**macOS / Linux:**
```bash
export GEMINI_API_KEY=your_key_here
```
 
### 4. Start the backend
 
```bash
cd backend
uvicorn main:app --reload --port 8000
```
 
You should see:
```
✓ orders:                3,421,083 rows
✓ order_products_prior: 32,434,489 rows
✓ order_products_train:  1,384,617 rows
✓ products:                 49,688 rows
✓ aisles:                     134 rows
✓ departments:                 21 rows
✓ order_products_all (prior + train): 33,819,106 rows
```
 
### 5. Start the frontend
 
```bash
cd frontend
npm install
npm run dev
```
 
Open **http://localhost:5173** in your browser.
 
---
