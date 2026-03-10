import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_retail_data(num_records=10000, num_customers=500):
    np.random.seed(42)
    random.seed(42)

    data = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Generate customer IDs, some buying frequently, some rarely
    customer_ids = np.random.choice(range(1, num_customers + 1), size=num_records, p=np.random.dirichlet(np.ones(num_customers)))

    for i in range(num_records):
        customer_id = customer_ids[i]
        
        # Determine frequency/monetary vibe based on customer ID to create natural clusters
        if customer_id < num_customers * 0.2: # High value, frequent
            quantity = np.random.randint(5, 50)
            unit_price = np.random.uniform(20.0, 150.0)
            date_offset = random.randint(0, 50) # Recent
        elif customer_id < num_customers * 0.6: # Mid value, average frequency
            quantity = np.random.randint(1, 20)
            unit_price = np.random.uniform(5.0, 50.0)
            date_offset = random.randint(0, 200)
        else: # Low value, infrequent, older
            quantity = np.random.randint(1, 5)
            unit_price = np.random.uniform(1.0, 20.0)
            date_offset = random.randint(200, 364)

        invoice_date = end_date - timedelta(days=date_offset)
        
        # Add a bit of random noise to date
        invoice_date += timedelta(hours=random.randint(0, 23), minutes=random.randint(0,59))

        invoice_no = f"INV{int(invoice_date.timestamp())}{random.randint(10,99)}"

        data.append({
            'InvoiceNo': invoice_no,
            'CustomerID': customer_id,
            'InvoiceDate': invoice_date,
            'UnitPrice': round(unit_price, 2),
            'Quantity': quantity,
            'Description': f"Product_{random.randint(1, 100)}"
        })

    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('InvoiceDate')
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/retail_data.csv', index=False)
    print("Generated synthetic retail dataset at data/retail_data.csv")

if __name__ == "__main__":
    generate_retail_data()
