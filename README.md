# Retail Customer Segmentation for Targeted Marketing

This project demonstrates how to perform customer segmentation using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering in Python to help a business personalize its marketing strategies.

## Features
- **Synthetic Data Generation**: Creates a realistic retail transaction dataset.
- **RFM Analysis**: Calculates how recently a customer bought, how often they buy, and how much they spend.
- **K-Means Clustering**: Uses scikit-learn to group customers into distinct segments.
- **Visualizations**: Produces scatter plots and boxplots to illustrate the segments to stakeholders.

## Setup

1. **Install Dependencies**: Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**: Run the script to create the synthetic dataset.
   ```bash
   python generate_data.py
   ```
   This will create `data/retail_data.csv`.

3. **Run Segmentation**: Execute the main script to process the data, perform clustering, and generate visual outputs.
   ```bash
   python segmentation.py
   ```

4. **Start the Dashboard**: Launch the interactive Streamlit dashboard to explore the grouped customers dynamically.
   ```bash
   streamlit run app.py
   ```

## Outputs

After running `segmentation.py`, check the `outputs/` folder for:
- `customer_segments.csv`: The final dataset with cluster assignments.
- `scatter_recency_monetary.png`: Visualization showing Recency vs. Monetary value.
- `scatter_frequency_monetary.png`: Visualization showing Frequency vs. Monetary value.
- `segment_boxplots.png`: Comparison of RFM distributions across segments.

## Segments Explained

Based on standard K-Means grouping and heuristic naming:
1. **High Value / Loyal**: High frequency, high monetary value, low recency. These are your best customers. Target them with early access or VIP rewards.
2. **Mid Value**: Average frequency and spending. Target them with cross-sell campaigns to increase their basket size.
3. **Low Value / At Risk**: High recency (haven't purchased in a long time), low frequency, low monetary value. Target them with re-engagement or discount campaigns.
