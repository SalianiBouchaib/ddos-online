import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="CSV Merger", page_icon="📊", layout="wide")

st.title("📊 CSV File Merger")
st.markdown("Upload multiple CSV files with the same columns to merge them into one file.")

# File uploader
uploaded_files = st. file_uploader(
    "Choose CSV files", 
    type="csv", 
    accept_multiple_files=True,
    help="Upload multiple CSV files that have the same column structure"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    try:
        # Read all CSV files
        dataframes = []
        file_info = []
        
        for uploaded_file in uploaded_files: 
            df = pd.read_csv(uploaded_file)
            dataframes. append(df)
            file_info.append({
                "filename": uploaded_file.name,
                "rows": len(df),
                "columns": list(df.columns)
            })
        
        # Display file information
        st.subheader("📁 Uploaded Files")
        for info in file_info:
            with st.expander(f"{info['filename']} - {info['rows']} rows"):
                st.write("**Columns:**", ", ".join(info['columns']))
        
        # Check if all files have the same columns
        first_columns = set(dataframes[0].columns)
        all_same_columns = all(set(df.columns) == first_columns for df in dataframes)
        
        if not all_same_columns:
            st.error("❌ Error: Not all files have the same columns!")
            st.write("**Column comparison:**")
            for i, info in enumerate(file_info):
                st.write(f"- {info['filename']}: {info['columns']}")
        else:
            st.success("✅ All files have matching columns!")
            
            # Merge dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Display merge summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(dataframes))
            with col2:
                st. metric("Total Rows", len(merged_df))
            with col3:
                st.metric("Columns", len(merged_df.columns))
            
            # Preview merged data
            st.subheader("👀 Preview Merged Data")
            st.dataframe(merged_df, use_container_width=True)
            
            # Download section
            st.subheader("⬇️ Download Merged File")
            
            col1, col2 = st. columns([2, 1])
            with col1:
                output_filename = st.text_input(
                    "Output filename", 
                    value="merged_data.csv",
                    help="Enter the name for your merged CSV file"
                )
            
            # Convert dataframe to CSV
            csv_buffer = StringIO()
            merged_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download Merged CSV",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.write("Please make sure all uploaded files are valid CSV files.")

else:
    st.info("👆 Please upload CSV files to get started")
    
    # Instructions
    st.markdown("""
    ### How to use:
    1. Click the "Browse files" button above
    2. Select multiple CSV files with the same column structure
    3. Review the preview of merged data
    4. Click "Download Merged CSV" to save the result
    
    ### Notes:
    - All CSV files must have the same columns
    - The merged file will contain all rows from all uploaded files
    - Row indices will be reset in the merged file
    """)