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

def determine_data_availability(match_percentage):
    """
    Determines data availability based on match percentage ranges:
    0-30%: High data gap (Not Available)
    30-50%: Partially Available
    50-100%: Fully Available
    """
    if match_percentage < 30:
        return "Not Available (High Data Gap)"
    elif match_percentage < 50:
        return "Partially Available"
    else:
        return "Fully Available"

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
                data_availability = determine_data_availability(similarity)
                matched_results.append({
                    "RowIndex": index,
                    "Definition": definition,
                    "Keyword": keyword,
                    "MatchPercentage": similarity,
                    "SustainabilityScore": sustainability_score,
                    "DataAvailability": data_availability
                })
                # Uncomment next line to keep only first match per definition:
                # break
    return matched_results

def match_excel_to_excel(df1, df2, col1="Definition", col2="Definition", threshold=70):
    """
    Compare definitions between two Excel files using fuzzy matching.
    Returns a list of dictionaries with match info if similarity >= threshold.
    """
    matched_results = []
    for idx1, row1 in df1.iterrows():
        definition1 = str(row1[col1])
        for idx2, row2 in df2.iterrows():
            definition2 = str(row2[col2])
            similarity = fuzz.token_set_ratio(definition1, definition2)
            if similarity >= threshold:
                sustainability_score = calculate_sustainability_score(similarity)
                data_availability = determine_data_availability(similarity)
                matched_results.append({
                    "Excel1_Row": idx1,
                    "Excel1_Definition": definition1,
                    "Excel2_Row": idx2,
                    "Excel2_Definition": definition2,
                    "MatchPercentage": similarity,
                    "SustainabilityScore": sustainability_score,
                    "DataAvailability": data_availability
                })
    return matched_results

def compute_semantic_similarity(definition, keyword):
    """
    Computes transformer-based semantic similarity between a definition and a keyword.
    """
    def_emb = semantic_model.encode(definition, convert_to_tensor=True)
    key_emb = semantic_model.encode(keyword, convert_to_tensor=True)
    cos_sim = util.cos_sim(def_emb, key_emb).item()  # value between 0 and 1
    return cos_sim * 100  # percentage

def compute_excel_semantic_similarity(definition1, definition2):
    """
    Computes transformer-based semantic similarity between two definitions.
    """
    def_emb1 = semantic_model.encode(definition1, convert_to_tensor=True)
    def_emb2 = semantic_model.encode(definition2, convert_to_tensor=True)
    cos_sim = util.cos_sim(def_emb1, def_emb2).item()  # value between 0 and 1
    return cos_sim * 100  # percentage

def main():
    st.title("Semantic Matching Tool")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["PDF to Excel", "Excel to Excel"])
    
    with tab1:
        st.header("PDF to Excel Matching")
        st.write("Upload your PDF file and Excel file (with a 'Definition' column) to perform semantic matching analysis.")

        # Add data availability explanation
        st.info("""
        **Data Availability Categories (Based on Semantic Match):**
        - **0-30%**: Not Available (High Data Gap)
        - **30-50%**: Partially Available
        - **50-100%**: Fully Available
        """)

        st.sidebar.header("PDF to Excel Configuration")
        pdf_fuzzy_threshold = st.sidebar.number_input("PDF-Excel Fuzzy Matching Threshold (%)", 
                                                 min_value=0, max_value=100, value=80, key="pdf_threshold")
        pdf_semantic_enabled = st.sidebar.checkbox("Compute PDF-Excel Semantic Similarity", value=True, key="pdf_semantic")
        pdf_excel_sheet = st.sidebar.text_input("Excel Sheet Name", value="Master Data POC", key="pdf_sheet")
        pdf_definition_column = st.sidebar.text_input("Definition Column Name", value="Definition", key="pdf_col")
        
        pdf_file = st.file_uploader("Upload PDF file", type=["pdf"], key="pdf_upload")
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="excel_upload")
        
        if pdf_file and excel_file:
            st.info("Files uploaded. Click the button below to start processing.")
            if st.button("Process PDF and Excel"):
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
                    df = analyze_excel_definitions(excel_file, sheet_name=pdf_excel_sheet, definition_column=pdf_definition_column)
                if df is None:
                    return
                st.write("### Excel Data Preview")
                st.dataframe(df.head())

                # Perform fuzzy matching
                with st.spinner("Performing fuzzy matching..."):
                    results = match_definitions_with_keywords(df, keywords, definition_column=pdf_definition_column, threshold=pdf_fuzzy_threshold)
                if not results:
                    st.warning("No fuzzy matches found over the specified threshold.")
                    return

                # Compute semantic similarity if enabled
                if pdf_semantic_enabled:
                    st.info("Computing semantic similarity scores...")
                    for match in results:
                        sem_score = compute_semantic_similarity(match["Definition"], match["Keyword"])
                        match["SemanticMatch"] = sem_score
                        # Data availability is now based solely on semantic match
                        match["DataAvailability"] = determine_data_availability(sem_score)
                else:
                    st.warning("Semantic similarity computation is required for data availability categorization. Enabling it automatically.")
                    pdf_semantic_enabled = True
                    for match in results:
                        sem_score = compute_semantic_similarity(match["Definition"], match["Keyword"])
                        match["SemanticMatch"] = sem_score
                        match["DataAvailability"] = determine_data_availability(sem_score)

                # Convert results to DataFrame and show the results table
                results_df = pd.DataFrame(results)
                st.write("### Matching Results")
                st.dataframe(results_df)

                # Display data availability summary
                if len(results_df) > 0:
                    st.write("### Data Availability Summary (Based on Semantic Match)")
                    availability_counts = results_df["DataAvailability"].value_counts()
                    st.write(availability_counts)
                    
                    # Create a pie chart for data availability
                    fig_pie, ax_pie = plt.subplots()
                    ax_pie.pie(availability_counts, labels=availability_counts.index, autopct='%1.1f%%', startangle=90)
                    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    st.pyplot(fig_pie)

                # Plot histogram of semantic similarity (if computed)
                fig, ax = plt.subplots()
                ax.hist(results_df["SemanticMatch"], bins=10, color='skyblue', edgecolor='black')
                ax.set_xlabel("Semantic Similarity (%)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Semantic Similarity Scores")
                st.pyplot(fig)

                # Compute overall PDF Score: average semantic similarity
                pdf_score = results_df["SemanticMatch"].mean()
                st.write(f"**PDF Score (Average Semantic Similarity): {pdf_score:.2f}%**")

                # Plot fuzzy match percentages
                fig2, ax2 = plt.subplots()
                ax2.hist(results_df["MatchPercentage"], bins=10, color='lightgreen', edgecolor='black')
                ax2.set_xlabel("Fuzzy Match Percentage")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Fuzzy Matching Scores")
                st.pyplot(fig2)

                # Option to download CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="pdf_excel_results.csv", mime="text/csv")

    with tab2:
        st.header("Excel to Excel Matching")
        st.write("Upload two Excel files to compare definitions between them.")
        
        # Add data availability explanation
        st.info("""
        **Data Availability Categories (Based on Semantic Match):**
        - **0-30%**: Not Available (High Data Gap)
        - **30-50%**: Partially Available
        - **50-100%**: Fully Available
        """)
        
        st.sidebar.header("Excel to Excel Configuration")
        excel_fuzzy_threshold = st.sidebar.number_input("Excel-Excel Fuzzy Matching Threshold (%)", 
                                                  min_value=0, max_value=100, value=80, key="excel_threshold")
        excel_semantic_enabled = st.sidebar.checkbox("Compute Excel-Excel Semantic Similarity", value=True, key="excel_semantic")
        
        col1, col2 = st.columns(2)
        
        with col1:
            excel1_file = st.file_uploader("Upload First Excel file", type=["xlsx", "xls"], key="excel1_upload")
            excel1_sheet = st.text_input("First Excel Sheet Name", value="Sheet1", key="excel1_sheet")
            excel1_column = st.text_input("First Excel Definition Column", value="Definition", key="excel1_col")
            
        with col2:
            excel2_file = st.file_uploader("Upload Second Excel file", type=["xlsx", "xls"], key="excel2_upload")
            excel2_sheet = st.text_input("Second Excel Sheet Name", value="Sheet1", key="excel2_sheet")
            excel2_column = st.text_input("Second Excel Definition Column", value="Definition", key="excel2_col")
        
        if excel1_file and excel2_file:
            st.info("Excel files uploaded. Click the button below to start processing.")
            if st.button("Compare Excel Files"):
                # Load Excel files
                with st.spinner("Loading Excel files..."):
                    df1 = analyze_excel_definitions(excel1_file, sheet_name=excel1_sheet, definition_column=excel1_column)
                    df2 = analyze_excel_definitions(excel2_file, sheet_name=excel2_sheet, definition_column=excel2_column)
                
                if df1 is None or df2 is None:
                    return
                
                st.write("### First Excel Data Preview")
                st.dataframe(df1.head())
                
                st.write("### Second Excel Data Preview")
                st.dataframe(df2.head())
                
                # Perform fuzzy matching between Excel files
                with st.spinner("Comparing Excel definitions..."):
                    results = match_excel_to_excel(df1, df2, col1=excel1_column, col2=excel2_column, threshold=excel_fuzzy_threshold)
                
                if not results:
                    st.warning("No matches found over the specified threshold.")
                    return
                
                # Compute semantic similarity if enabled
                if excel_semantic_enabled:
                    st.info("Computing semantic similarity scores...")
                    for match in results:
                        sem_score = compute_excel_semantic_similarity(match["Excel1_Definition"], match["Excel2_Definition"])
                        match["SemanticMatch"] = sem_score
                        # Data availability is now based solely on semantic match
                        match["DataAvailability"] = determine_data_availability(sem_score)
                else:
                    st.warning("Semantic similarity computation is required for data availability categorization. Enabling it automatically.")
                    excel_semantic_enabled = True
                    for match in results:
                        sem_score = compute_excel_semantic_similarity(match["Excel1_Definition"], match["Excel2_Definition"])
                        match["SemanticMatch"] = sem_score
                        match["DataAvailability"] = determine_data_availability(sem_score)
                
                # Convert results to DataFrame and show the results table
                results_df = pd.DataFrame(results)
                st.write("### Matching Results")
                st.dataframe(results_df)
                
                # Display data availability summary
                if len(results_df) > 0:
                    st.write("### Data Availability Summary (Based on Semantic Match)")
                    availability_counts = results_df["DataAvailability"].value_counts()
                    st.write(availability_counts)
                    
                    # Create a pie chart for data availability
                    fig_pie, ax_pie = plt.subplots()
                    ax_pie.pie(availability_counts, labels=availability_counts.index, autopct='%1.1f%%', startangle=90)
                    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    st.pyplot(fig_pie)
                
                # Plot histogram of semantic similarity
                fig, ax = plt.subplots()
                ax.hist(results_df["SemanticMatch"], bins=10, color='skyblue', edgecolor='black')
                ax.set_xlabel("Semantic Similarity (%)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Semantic Similarity Scores")
                st.pyplot(fig)
                
                # Compute overall Excel Score: average semantic similarity
                excel_score = results_df["SemanticMatch"].mean()
                st.write(f"**Excel Comparison Score (Average Semantic Similarity): {excel_score:.2f}%**")
                
                # Plot fuzzy match percentages
                fig2, ax2 = plt.subplots()
                ax2.hist(results_df["MatchPercentage"], bins=10, color='lightgreen', edgecolor='black')
                ax2.set_xlabel("Fuzzy Match Percentage")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Fuzzy Matching Scores")
                st.pyplot(fig2)
                
                # Option to download CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="excel_excel_results.csv", mime="text/csv")

# This is the correct way to run a Streamlit app
if __name__ == "__main__":
    # Check if the script is being run directly with Python
    if 'STREAMLIT_RUN_APP' not in os.environ:
        # Display a message instructing the user how to run the app properly
        print("To run this Streamlit app, use the command:")
        print(f"streamlit run {os.path.basename(__file__)}")
        print("\nStarting Streamlit server for you...")
        
        # Set an environment variable to prevent recursive execution
        os.environ['STREAMLIT_RUN_APP'] = '1'
        
        # Run the Streamlit command once
        import subprocess
        file_path = os.path.abspath(__file__)
        subprocess.run(["streamlit", "run", file_path])
    else:
        # If already running through Streamlit, just execute main()
        main()