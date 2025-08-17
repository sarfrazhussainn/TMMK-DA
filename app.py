import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import os
import tempfile
import time
try:
    from google import genai
except ImportError:
    import google.generativeai as genai
import json
import plotly.graph_objects as go
import hashlib

# Remove Windows-specific Tesseract path - not needed on Linux cloud servers
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---- Configuration ----
st.set_page_config(page_title="Voter Data Analysis", layout="wide")

# ---- PDF Processing Functions (from voterr.py) ----
def extract_text_from_box(image):
    """Run OCR on a cropped box."""
    try:
        return pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def parse_voter_box(text):
    """Extract only Name and Relative Name from OCR text."""
    data = {
        "Name": "",
        "Relative Name": ""
    }

    # Name
    m = re.search(r"Name\s*:\s*([^\n-]+)", text)
    if m:
        data["Name"] = m.group(1).strip()

    # Relative Name (Father/Husband/Other)
    m = re.search(r"(Father|Husband|Other)\s*Name\s*:\s*([^\n-]+)", text)
    if m:
        data["Relative Name"] = m.group(2).strip()

    return data

def merge_boxes(boxes):
    """Merge overlapping or very close boxes."""
    merged = []
    for b in boxes:
        added = False
        for i, m in enumerate(merged):
            if (abs(b[0] - m[0]) < 20 and abs(b[1] - m[1]) < 20) or \
               (b[0] < m[0] + m[2] and b[0] + b[2] > m[0] and b[1] < m[1] + m[3] and b[1] + b[3] > m[1]):
                # Merge into one rectangle
                x = min(b[0], m[0])
                y = min(b[1], m[1])
                w = max(b[0] + b[2], m[0] + m[2]) - x
                h = max(b[1] + b[3], m[1] + m[3]) - y
                merged[i] = (x, y, w, h)
                added = True
                break
        if not added:
            merged.append(b)
    return merged

def detect_boxes(page_img, page_num=1):
    """Detect main voter boxes using OpenCV with merging."""
    try:
        img_cv = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Invert & threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get page size for scaling detection
        ph, pw = gray.shape
        min_w, max_w = pw * 0.15, pw * 0.40
        min_h, max_h = ph * 0.04, ph * 0.12

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if min_w < w < max_w and min_h < h < max_h:
                boxes.append((x, y, w, h))

        # Merge boxes to avoid splits
        boxes = merge_boxes(sorted(boxes, key=lambda b: (b[1], b[0])))

        # Sort top-to-bottom, left-to-right
        boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))

        return boxes
    except Exception as e:
        st.error(f"Box detection error: {e}")
        return []

def process_pdf_opencv(pdf_file, dpi=300, padding=10):
    """Process PDF using OpenCV approach from voterr.py"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        pages = convert_from_path(tmp_file_path, dpi=dpi)
        all_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for page_num, page in enumerate(pages, start=1):
            status_text.text(f"Processing page {page_num}/{len(pages)}...")
            
            boxes = detect_boxes(page, page_num)

            for (x, y, w, h) in boxes:
                # Add padding to avoid cutting text
                crop_box = (
                    max(0, x - padding),
                    max(0, y - padding),
                    x + w + padding,
                    y + h + padding
                )
                cropped = page.crop(crop_box)
                text = extract_text_from_box(cropped)

                if text:
                    parsed = parse_voter_box(text)
                    # Only add if at least one name is found
                    if parsed["Name"] or parsed["Relative Name"]:
                        all_data.append(parsed)
            
            progress_bar.progress(page_num / len(pages))

        progress_bar.empty()
        status_text.empty()
        
        return all_data

    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return []
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

# ---- Muslim Name Analysis Functions ----
def extract_muslim_names(names, client):
    """Extract Muslim names from a list of names using Gemini API"""
    prompt = (
        "Here is a list of names:\n"
        + "\n".join(names)
        + "\n\nAnalyze each name and identify which ones are Muslim names. "
        "Consider variations in spelling and transliterations. "
        "Respond with only a JSON array of the names that are Muslim names. "
        "Example format: [\"Ahmed\", \"Fatima\", \"Hassan\"]"
    )
    
    try:
        # Try the new API structure first
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(prompt)
            text = resp.text.strip()
        except:
            # Fallback to old structure
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            text = resp.text.strip()
        
        # Clean the response text
        if text.startswith('```json'):
            text = text.replace('```json', '').replace('```', '').strip()
        elif text.startswith('```'):
            text = text.replace('```', '').strip()
        
        # Try to parse as JSON
        try:
            parsed_data = json.loads(text)
            if isinstance(parsed_data, list):
                return parsed_data
            else:
                return []
        except (json.JSONDecodeError, ValueError):
            # Extract names from quotes
            matches = re.findall(r'"([^"]+)"', text)
            return matches if matches else []
    except Exception as e:
        st.error(f"Error processing batch: {e}")
        return []

def process_names_for_analysis(names, client, batch_size=500, pause=1.0):
    """Process names in batches for Muslim name analysis"""
    total_muslim_names = []
    batch_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_count += 1
        
        status_text.text(f"Processing batch {batch_count}/{(len(names)-1)//batch_size + 1}...")
        
        muslim_names_batch = extract_muslim_names(batch, client)
        total_muslim_names.extend(muslim_names_batch)
        
        progress = (i + batch_size) / len(names)
        progress_bar.progress(min(progress, 1.0))
        
        if i + batch_size < len(names):
            time.sleep(pause)
    
    progress_bar.empty()
    status_text.empty()
    
    return total_muslim_names

# ---- Visualization Functions ----
def create_comparison_chart(extracted_df, muslim_names):
    """Create comparison chart"""
    total_names = len(extracted_df) if extracted_df is not None and not extracted_df.empty else 0
    total_muslim_names = len(muslim_names) if muslim_names else 0
    non_muslim_names = total_names - total_muslim_names
    
    fig = go.Figure(data=[go.Bar(
        x=['Total Names', 'Muslim Names', 'Non-Muslim Names'],
        y=[total_names, total_muslim_names, non_muslim_names],
        marker_color=['#1f77b4', '#2E8B57', '#FF6B6B'],
        text=[total_names, total_muslim_names, non_muslim_names],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Name Analysis Summary",
        xaxis_title="Categories",
        yaxis_title="Count",
        title_x=0.5,
        font=dict(size=14),
        height=400
    )
    
    return fig

def create_pie_chart(extracted_df, muslim_names):
    """Create pie chart"""
    total_names = len(extracted_df) if extracted_df is not None and not extracted_df.empty else 0
    total_muslim_names = len(muslim_names) if muslim_names else 0
    non_muslim_names = total_names - total_muslim_names
    
    if total_names == 0:
        # Create empty pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['No Data'],
            values=[1],
            hole=0.3,
            marker_colors=['#CCCCCC']
        )])
    else:
        fig = go.Figure(data=[go.Pie(
            labels=['Muslim Names', 'Non-Muslim Names'],
            values=[total_muslim_names, non_muslim_names],
            hole=0.3,
            marker_colors=['#2E8B57', '#FF6B6B'],
            textinfo='label+percent+value',
            textfont_size=12
        )])
    
    fig.update_layout(
        title="Distribution of Muslim vs Non-Muslim Names",
        title_x=0.5,
        font=dict(size=14),
        height=400
    )
    
    return fig

# ---- Main Streamlit App ----
def main():
    st.title("Voter Data Analysis")
    st.markdown("Extract names from voter PDFs, analyze Muslim names, and visualize data")
    st.markdown("---")
    
    # Add system info for debugging
    if st.sidebar.checkbox("Show System Info", False):
        st.sidebar.write("**System Information:**")
        st.sidebar.write(f"OpenCV version: {cv2.__version__}")
        st.sidebar.write(f"Streamlit version: {st.__version__}")
        try:
            st.sidebar.write(f"Tesseract available: {pytesseract.get_tesseract_version()}")
        except:
            st.sidebar.write("Tesseract: Not accessible")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # PDF processing parameters
    st.sidebar.subheader("PDF Processing Parameters")
    dpi = st.sidebar.number_input("DPI (Image Resolution)", value=300, min_value=150, max_value=600, step=50)
    padding = st.sidebar.number_input("Box Padding", value=10, min_value=5, max_value=30)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    batch_size = st.sidebar.slider("Batch Size", min_value=100, max_value=1000, value=500, step=100)
    pause_time = st.sidebar.slider("Pause Between Batches (seconds)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 PDF Processing", "📊 Muslim Name Analysis", "📈 Data Visualization"])
    
    with tab1:
        st.header("PDF Voter Data Extraction")
        st.info("Upload a PDF file containing voter data to extract names and relative names using OpenCV detection.")
        
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"File uploaded: {uploaded_file.name}")
            st.info(f"File size: {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
            
            # Check if this is a new file
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            if 'current_file_hash' not in st.session_state or st.session_state.current_file_hash != file_hash:
                # Clear previous analysis results when new file is uploaded
                if 'muslim_names' in st.session_state:
                    del st.session_state.muslim_names
                if 'muslim_df' in st.session_state:
                    del st.session_state.muslim_df
                if 'extracted_data' in st.session_state:
                    del st.session_state.extracted_data
                if 'extracted_df' in st.session_state:
                    del st.session_state.extracted_df
                st.session_state.current_file_hash = file_hash
                st.info("🔄 New file detected. Previous analysis results cleared.")
            
            if st.button("🔍 Process PDF", type="primary"):
                with st.spinner("Processing PDF with OpenCV detection..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        extracted_data = process_pdf_opencv(uploaded_file, dpi=dpi, padding=padding)
                        
                        if extracted_data:
                            st.success(f"✅ PDF processed successfully! Found {len(extracted_data)} name records.")
                            
                            # Create DataFrame
                            df = pd.DataFrame(extracted_data)
                            
                            # Store in session state
                            st.session_state.extracted_data = extracted_data
                            st.session_state.extracted_df = df
                            st.session_state.current_file_name = uploaded_file.name
                            
                            # Display data preview
                            st.subheader("📊 Extracted Data Preview")
                            st.dataframe(df, height=400)
                            
                            # Download CSV
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Extracted Data CSV",
                                csv_data,
                                file_name="voter_extracted_data.csv",
                                mime="text/csv"
                            )
                            
                            # Display statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", len(df))
                            with col2:
                                valid_names = len(df[df['Name'].notna() & (df['Name'] != '')])
                                st.metric("Valid Names", valid_names)
                            with col3:
                                valid_relatives = len(df[df['Relative Name'].notna() & (df['Relative Name'] != '')])
                                st.metric("Valid Relative Names", valid_relatives)
                        
                        else:
                            st.error("❌ No data could be extracted from the PDF. Please check the file format.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
        
        else:
            st.info("👆 Please upload a PDF file to get started.")
    
    with tab2:
        st.header("Muslim Name Analysis")
        
        # API Key input
        api_key = st.text_input("Google Gemini API Key", type="password", 
                               help="Enter your Google Gemini API key for Muslim name analysis")
        
        if 'extracted_df' not in st.session_state:
            st.warning("⚠️ Please process a PDF first in the PDF Processing tab.")
            return
        
        # Display current file info
        current_file = st.session_state.get('current_file_name', 'Unknown')
        st.info(f"📄 Current file: {current_file}")
        
        if not api_key:
            st.warning("⚠️ Please enter your Google Gemini API key above.")
            return
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Test the client with a simple message
            test_response = model.generate_content("Test")
            st.success("✅ Gemini API connection successful!")
            client = genai  # For compatibility with existing code
        except Exception as e:
            st.error(f"❌ Error initializing Gemini client: {e}")
            return
        
        df = st.session_state.extracted_df
        
        st.subheader("📊 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Names", len(df))
        with col2:
            valid_names = len(df[df['Name'].notna() & (df['Name'] != '')])
            st.metric("Valid Names for Analysis", valid_names)
        
        # Check if analysis already done for current file
        analysis_done = ('muslim_names' in st.session_state and 
                        'muslim_df' in st.session_state and
                        'current_file_hash' in st.session_state)
        
        # Add a "Reset Analysis" button
        if analysis_done:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Reset Analysis", help="Clear current analysis to re-analyze"):
                    if 'muslim_names' in st.session_state:
                        del st.session_state.muslim_names
                    if 'muslim_df' in st.session_state:
                        del st.session_state.muslim_df
                    st.rerun()
        
        if analysis_done:
            st.info("✅ Muslim name analysis already completed.")
            muslim_names = st.session_state.muslim_names
            muslim_df = st.session_state.muslim_df
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Muslim Names Found", len(muslim_names))
            with col2:
                percentage = (len(muslim_names) / valid_names) * 100 if valid_names > 0 else 0
                st.metric("Muslim Percentage", f"{percentage:.1f}%")
            with col3:
                st.metric("Non-Muslim Names", valid_names - len(muslim_names))
            
            # Display Muslim names
            if not muslim_df.empty:
                st.subheader("Muslim Names Found")
                st.dataframe(muslim_df, height=300)
                
                # Download Muslim names CSV
                muslim_csv = muslim_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Muslim Names CSV",
                    muslim_csv,
                    file_name="muslim_names.csv",
                    mime="text/csv"
                )
        
        if st.button("🔍 Analyze Muslim Names", type="primary", disabled=analysis_done):
            names = df['Name'].dropna().astype(str).tolist()
            # Filter out empty names
            names = [name.strip() for name in names if name.strip()]
            
            if not names:
                st.error("❌ No valid names found for analysis.")
                return
            
            with st.spinner("Analyzing names with Gemini AI..."):
                try:
                    muslim_names = process_names_for_analysis(names, client, batch_size, pause_time)
                    
                    # Create Muslim names DataFrame
                    muslim_df = pd.DataFrame({"Name": muslim_names}) if muslim_names else pd.DataFrame()
                    
                    # Store in session state
                    st.session_state.muslim_names = muslim_names
                    st.session_state.muslim_df = muslim_df
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Muslim Names Found", len(muslim_names))
                    with col2:
                        percentage = (len(muslim_names) / len(names)) * 100 if names else 0
                        st.metric("Muslim Percentage", f"{percentage:.1f}%")
                    with col3:
                        st.metric("Non-Muslim Names", len(names) - len(muslim_names))
                    
                    # Display Muslim names
                    if muslim_names:
                        st.subheader("Muslim Names Found")
                        st.dataframe(muslim_df, height=300)
                        
                        # Download Muslim names CSV
                        muslim_csv = muslim_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Muslim Names CSV",
                            muslim_csv,
                            file_name="muslim_names.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No Muslim names were identified in the dataset.")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    with tab3:
        st.header("Data Visualization")
        
        if 'extracted_df' not in st.session_state:
            st.warning("⚠️ Please process a PDF first in the PDF Processing tab.")
            return
        
        # Display current file info
        current_file = st.session_state.get('current_file_name', 'Unknown')
        st.info(f"📄 Visualizing data from: {current_file}")
        
        extracted_df = st.session_state.extracted_df
        muslim_names = st.session_state.get('muslim_names', [])
        muslim_df = st.session_state.get('muslim_df', pd.DataFrame())
        
        if extracted_df.empty:
            st.warning("No data available for visualization.")
            return
        
        st.subheader("📊 Data Overview")
        
        # Metrics
        total_names = len(extracted_df)
        valid_names = len(extracted_df[extracted_df['Name'].notna() & (extracted_df['Name'] != '')])
        muslim_count = len(muslim_names) if muslim_names else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", total_names)
        with col2:
            st.metric("Valid Names", valid_names)
        with col3:
            st.metric("Muslim Names", muslim_count)
        with col4:
            percentage = (muslim_count / valid_names) * 100 if valid_names > 0 else 0
            st.metric("Muslim %", f"{percentage:.1f}%")
        
        # Visualizations
        if muslim_names:  # Only show charts if Muslim analysis is done
            col1, col2 = st.columns(2)
            
            with col1:
                comparison_fig = create_comparison_chart(extracted_df, muslim_names)
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                pie_fig = create_pie_chart(extracted_df, muslim_names)
                st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("📊 Complete the Muslim name analysis to see detailed visualizations.")
        
        # Data tables
        st.subheader("📋 Data Tables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**All Extracted Names**")
            st.dataframe(extracted_df, height=400)
        
        with col2:
            if not muslim_df.empty:
                st.write("**Muslim Names**")
                st.dataframe(muslim_df, height=400)
            else:
                st.info("Complete Muslim name analysis to see Muslim names here.")
        
        # Download section
        st.subheader("📥 Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if not extracted_df.empty:
                extracted_csv = extracted_df.to_csv(index=False)
                st.download_button(
                    "📥 Download All Extracted Names",
                    extracted_csv,
                    file_name="all_extracted_names.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not muslim_df.empty:
                muslim_csv = muslim_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Muslim Names Only",
                    muslim_csv,
                    file_name="muslim_names_only.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **Made with Streamlit** | "
        "💡 **Powered by Google Gemini AI** | "
        "📄 **OpenCV PDF Processing** | "
        "🔍 **Tesseract OCR**"
    )

if __name__ == "__main__":
    main()
