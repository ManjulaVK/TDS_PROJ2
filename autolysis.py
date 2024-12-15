
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from tabulate import tabulate
import requests
from rich.pretty import pprint


from dotenv import load_dotenv
load_dotenv()
# Access the AIPROXY_TOKEN from the environment variables
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

url="http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
model="gpt-4o-mini"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}



# Function to analyze metadata of the CSV
def analyze_metadata(df):
    """Analyze and summarize metadata of the CSV."""
    metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "column_types": df.dtypes.apply(str).to_dict(),
        "example_values": df.head(1).to_dict(orient="records")[0],
    }
    return metadata

# Function to interact with GPT-4o-Mini via AI Proxy
def query_llm(prompt):
    """Query GPT-4o-Mini via AI Proxy."""
    token = os.environ["AIPROXY_TOKEN"]
    response = requests.post(
        "https://api.aiproxy.com/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "gpt-4o-mini", "prompt": prompt},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Function to generate correlation heatmap
def generate_correlation_heatmap(df, filename="chart1.png"):
    """Generate and save a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(filename)
    plt.close()

# Function to generate distribution plot
def generate_distribution_plot(df, column, filename="chart2.png"):
    """Generate and save a distribution plot."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

# Function to generate clustering visualization
def generate_clustering_plot(df, filename="chart3.png"):
    """Generate and save a clustering visualization."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.select_dtypes(include=np.number).dropna())

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced_data)
    labels = kmeans.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50)
    plt.title("Clustering Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(filename)
    plt.close()

# Main function
def main():
    # Get filename from command-line arguments or Colab configuration
    input_file = "/content/drive/MyDrive/happiness.csv"  # Change this to your uploaded file path

    # Load dataset
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Ensure it is uploaded to the right path.")
        sys.exit(1)

    # Step 1: Analyze metadata
    metadata = analyze_metadata(df)
    pprint(metadata)

    # Step 2: Summarize data and send to GPT-4o-Mini
    metadata_summary = (
        f"The dataset has {metadata['num_rows']} rows and {metadata['num_columns']} columns. "
        f"Missing values: {metadata['missing_values']}. Column types: {metadata['column_types']}."
    )
    llm_prompt = f"""
    Here is a dataset summary:
    {metadata_summary}

    Suggest the best analysis, visualizations, and insights we can generate from this dataset.
    Provide Python code for any advanced operations.
    """
    llm_response = query_llm(llm_prompt)
    print("GPT-4o-Mini Response:")
    pprint(llm_response)

    # Step 3: Perform visualizations
    # Generate Correlation Heatmap
    generate_correlation_heatmap(df)

    # Generate Distribution Plot (on first numeric column)
    numeric_columns = df.select_dtypes(include=np.number).columns
    if len(numeric_columns) > 0:
        generate_distribution_plot(df, numeric_columns[0])

    # Generate Clustering Visualization
    if len(numeric_columns) > 1:
        generate_clustering_plot(df)

    # Step 4: Narrate story using GPT-4o-Mini
    narrative_prompt = f"""
    Based on the dataset analysis and visualizations, write a narrative story that:
    1. Briefly describes the dataset.
    2. Summarizes the key insights and findings.
    3. Explains the implications of the findings.
    Include references to the generated visualizations.
    """
    story = query_llm(narrative_prompt)

    # Save narrative as README.md
    with open("README.md", "w") as readme_file:
        readme_file.write("# Automated Data Analysis and Story

")
        readme_file.write(story)
        readme_file.write("

## Visualizations
")
        readme_file.write("1. Correlation Heatmap: ![Correlation Heatmap](chart1.png)
")
        if len(numeric_columns) > 0:
            readme_file.write("2. Distribution Plot: ![Distribution Plot](chart2.png)
")
        if len(numeric_columns) > 1:
            readme_file.write("3. Clustering Plot: ![Clustering Plot](chart3.png)
")

    print("Analysis complete. Results saved in README.md and PNG files.")

if __name__ == "__main__":
    main()
