import os
import re
import sys
import io
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt

from PyPDF2 import PdfReader
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

# Load transformer model (GPU enabled if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def extract_keywords_from_pdf(file, max_line_length=200):
    """
    Extract potential keywords from a PDF file object.
    """
    try:
        reader = PdfReader(file)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []
    all_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    keywords = []
    for line in all_text.splitlines():
        cleaned = line.strip()
        if cleaned and len(cleaned) <= max_line_length:
            if re.search("[A-Za-z]", cleaned):
                keywords.append(cleaned)
    unique_keywords = list(set(keywords))
    return unique_keywords

def analyze_excel_definitions(file, sheet_name="Sheet1", definition_column="Definition"):
    """
    Reads an Excel file object and returns a DataFrame containing the definition column.
    """
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    if definition_column not in df.columns:
        st.error(f"Error: Column '{definition_column}' not found in Excel sheet.")
        return None
    return df

def calculate_sustainability_score(match_percentage):
    """
    In this example, the sustainability score is equal to the match percentage.
    """
    return match_percentage

def match_definitions_with_keywords(df, keywords, definition_column="Definition", threshold=70):
    """
    Compare each definition with keywords using fuzzy matching.
    Returns a list of dictionaries with match info if similarity >= threshold.
    """
    matched_results = []
    for index, row in df.iterrows():
        definition = str(row[definition_column])
        for keyword in keywords:
            similarity = fuzz.token_set_ratio(definition, keyword)
            if similarity >= threshold:
                sustainability_score = calculate_sustainability_score(similarity)
                matched_results.append({
                    "RowIndex": index,
                    "Definition": definition,
                    "Keyword": keyword,
                    "MatchPercentage": similarity,
                    "SustainabilityScore": sustainability_score
                })
                # Uncomment next line to keep only first match per definition:
                # break
    return matched_results

def compute_semantic_similarity(definition, keyword):
    """
    Computes transformer-based semantic similarity between a definition and a keyword.
    """
    def_emb = semantic_model.encode(definition, convert_to_tensor=True)
    key_emb = semantic_model.encode(keyword, convert_to_tensor=True)
    cos_sim = util.cos_sim(def_emb, key_emb).item()  # value between 0 and 1
    return cos_sim * 100  # percentage

def main():
    st.title("Semantic Matching with PDF and Excel")
    st.write("Upload your PDF file and Excel file (with a 'Definition' column) to perform semantic matching analysis.")

    st.sidebar.header("Configuration")
    fuzzy_threshold = st.sidebar.number_input("Fuzzy Matching Threshold (%)", min_value=0, max_value=100, value=80)
    semantic_enabled = st.sidebar.checkbox("Compute Semantic Similarity", value=True)
    excel_sheet = st.sidebar.text_input("Excel Sheet Name", value="Master Data POC")
    definition_column = st.sidebar.text_input("Definition Column Name", value="Definition")
    
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if pdf_file and excel_file:
        st.info("Files uploaded. Click the button below to start processing.")
        if st.button("Process Files"):
            # Extract keywords from PDF
            with st.spinner("Extracting keywords from PDF..."):
                keywords = extract_keywords_from_pdf(pdf_file)
            if not keywords:
                st.error("No keywords extracted from PDF.")
                return
            st.write(f"**Extracted Keywords ({len(keywords)}):**")
            st.write(keywords)
            
            # Load Excel definitions
            with st.spinner("Loading Excel definitions..."):
                df = analyze_excel_definitions(excel_file, sheet_name=excel_sheet, definition_column=definition_column)
            if df is None:
                return
            st.write("### Excel Data Preview")
            st.dataframe(df.head())

            # Perform fuzzy matching
            with st.spinner("Performing fuzzy matching..."):
                results = match_definitions_with_keywords(df, keywords, definition_column=definition_column, threshold=fuzzy_threshold)
            if not results:
                st.warning("No fuzzy matches found over the specified threshold.")
                return

            # Compute semantic similarity if enabled
            if semantic_enabled:
                st.info("Computing semantic similarity scores...")
                for match in results:
                    sem_score = compute_semantic_similarity(match["Definition"], match["Keyword"])
                    match["SemanticMatch"] = sem_score
            else:
                for match in results:
                    match["SemanticMatch"] = None

            # Convert results to DataFrame and show the results table
            results_df = pd.DataFrame(results)
            st.write("### Matching Results")
            st.dataframe(results_df)

            # Plot histogram of semantic similarity (if computed)
            if semantic_enabled:
                fig, ax = plt.subplots()
                ax.hist(results_df["SemanticMatch"], bins=10, color='skyblue', edgecolor='black')
                ax.set_xlabel("Semantic Similarity (%)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Semantic Similarity Scores")
                st.pyplot(fig)

                # Compute overall PDF Score: average semantic similarity
                pdf_score = results_df["SemanticMatch"].mean()
                st.write(f"**PDF Score (Average Semantic Similarity): {pdf_score:.2f}%**")
            else:
                st.write("Semantic similarity not computed.")

            # Plot fuzzy match percentages
            fig2, ax2 = plt.subplots()
            ax2.hist(results_df["MatchPercentage"], bins=10, color='lightgreen', edgecolor='black')
            ax2.set_xlabel("Fuzzy Match Percentage")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of Fuzzy Matching Scores")
            st.pyplot(fig2)

            # Option to download CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="matched_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()