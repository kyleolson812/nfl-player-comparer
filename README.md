# 🏈 NFL Player Comparison API

A FastAPI backend that compares NFL wide receivers based on their most recent season stats using machine learning similarity metrics (cosine similarity).

## 🚀 Features

- Compare a given NFL WR to others based on performance
- Uses real player data from [nfl_data_py](https://github.com/nflverse/nfl_data_py)
- Standardized and normalized stats
- FastAPI-powered REST API

## 📦 Requirements

- Python 3.8+
- pip install -r requirements.txt

## 📁 Project Structure

nfl-player-comparer/
├── app.py # FastAPI app
├── model/
│ ├── init.py
│ └── similarity_engine.py # ML logic and data loading
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Getting Started

1. **Clone the Repo**

   ```bash
   git clone https://github.com/your-username/nfl-player-comparer.git
   cd nfl-player-comparer

   ```

2. **Install Dependencies**
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Run the Server**
   uvicorn app:app --reload

4. Test the API
   Go to http://localhost:8000/docs for SwaggerUI

## 📈 Example Request

GET /similar?player=Justin Jefferson&top_k=5

[
"Tyreek Hill",
"Amon-Ra St. Brown",
"Stefon Diggs",
"CeeDee Lamb",
"Keenan Allen"
]

📊 How It Works
Loads current-year WR data from nfl_data_py

Filters and standardizes stats (targets, receptions, yards, TDs)

Uses cosine similarity to compare players' performance profiles

Exposes a simple /similar API to find the closest matches

🔮 Future Ideas
Add support for RBs, QBs, TEs

Add clustering with UMAP or PCA for visualization

Frontend for player selection and comparison graphs

Deploy to Railway, Render, or Hugging Face Spaces

📘 Credit
nfl_data_py

scikit-learn

FastAPI
