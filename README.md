# newsvendor-theft-simulation
Stochastic Newsvendor model with theft-adjusted demand simulation and Poisson-based expected profit optimization.
# Newsvendor Model with Theft — Simulation + Optimization

This project analyzes how **inventory theft (shrinkage)** affects the optimal order quantity in the classic **Newsvendor Problem**.

We model:
- Customer arrivals as **Poisson(λ)**
- Theft events as **Poisson(λ × thief%)**
- Inventory consumption = **sales + stolen units**
- Profit = revenue – purchasing cost + salvage value on leftovers

The goal is to find the **optimal starting inventory Q\*** that **maximizes expected profit** under different:
- Theft percentages
- Season lengths
- Cost and salvage rates

---

## 📁 Repository Structure

newsvendor-theft-simulation/
│
├─ src/ # Python code scripts
│ ├─ simulation_model.py
│ └─ expected_profit_poisson.py (optional placeholder)
│
├─ notebooks/ # Jupyter notebooks for experiments
│ └─ NVMS_5_May_v2.ipynb
│
├─ figures/ # (optional) plots will be saved here
│
└─ data/ # (empty) no real dataset is needed



> **Note:** This model is **fully simulation-based**.  
> No external dataset is required.  
> All demand and theft values are generated using Poisson random sampling.



## 🧠 Key Idea

Because theft reduces inventory **before** customers arrive, the **observed demand is distorted**.  
If we ignore theft, we **underestimate true demand** and order too little.

This is important in:
- Retail store shrinkage
- Warehouse pilferage
- Supply chain safety stock planning

---

---
## ▶️ How to Run (Beginner-Friendly)

You can run this in **Google Colab** (no installation needed).

1. Open Colab: https://colab.research.google.com
2. Upload the notebook from the `notebooks/` folder
3. Run the cells in order
4. The model will compute:
   - Optimal Q\*
   - Expected profit
   - Demand / theft statistics

---
## 👤 Author

**Md Shoaib Ullash**  
Graduate Research Assistant  
Texas State University  
Email: imf42@txstate.edu  
GitHub: https://github.com/ullash-s

