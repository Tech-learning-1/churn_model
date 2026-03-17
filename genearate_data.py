import pandas as pd
import numpy as np
import os
from pathlib import Path
np.random.seed(42)

n_samples = 1000

data = {
       'custormer_id' : range(1,n_samples + 1),
       'age' : np.random.randint(18,72,n_samples),
       'tenure_months' : np.random.randint(1,72,n_samples),
       'monthly_charges' : np.random.randint(20, 120, n_samples),
       'Total_charges': np.random.randint(100, 8000, n_samples),
       'support_calls' : np.random.randint(0, 10, n_samples)
}

churnprob = ((1 - data['tenure_months']/72 * 0.3) + (data ['monthly_charges']/120 * 0.3) + (data['support_calls']/10 * 0.4))

data ['churn'] = (np.random.random(n_samples) < churnprob).astype(int)


folder = Path("data")
if not folder.exists():
    folder.mkdir(exist_ok=True)

csv_file = folder / "churn_data.csv"
if not csv_file.exists():
     folder.touch()

df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)
print("Generated", len(df) , "samples")
print(f"churn rate : {df['churn'].mean() : .2}")