import streamlit as st
import os
import pandas as pd
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.units import inch
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="MediScan Pro - Medical Report Analyzer",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Custom CSS for creamish white theme with linear layout
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #f8f5f0;  /* Creamish white background */
        font-family: 'Inter', sans-serif;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Header */
    .main-header {
        background: #d4b483;  /* Warm cream */
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: #5a4a3a;  /* Dark brown */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #5a4a3a;  /* Dark brown */
        text-align: center;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Cards */
    .analysis-card {
        background: #fffdfa;  /* Off-white */
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8e0d0;  /* Light cream border */
    }
    
    /* Input Section */
    .input-section {
        background: #fffdfa;  /* Off-white */
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8e0d0;
    }
    
    /* Status Indicators */
    .emergency-alert {
        background: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .normal-status {
        background: #a8c69f;  /* Soft green */
        color: #2d4a2a;  /* Dark green */
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f1e9dd;  /* Light cream */
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d4b483;
        background: #fffdfa;
    }
    
    /* Buttons */
    .stButton > button {
        background: #d4b483;  /* Warm cream */
        color: #5a4a3a;  /* Dark brown */
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #c4a473;
        color: #5a4a3a;
    }
    
    /* Section Headers */
    .section-header {
        background: #d4b483;  /* Warm cream */
        color: #5a4a3a;  /* Dark brown */
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding: 1rem;
    }
    
    /* Status text colors */
    .status-normal {
        color: #2d4a2a;  /* Dark green */
    }
    
    .status-warning {
        color: #b58a3a;  /* Gold */
    }
    
    .status-critical {
        color: #a83a3a;  /* Dark red */
    }
    
    .status-emergency {
        color: #d63031;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API Keys
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Check if API keys are available
if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    st.error("🔑 API keys are missing. Please check your configuration.")
    st.stop()

MAX_IMAGE_WIDTH = 600  # Slightly larger for single column layout

# (Keep all the existing functions: get_medical_agent, get_comprehensive_agent, 
# resize_image_for_display, analyze_medical_scan, comprehensive_health_assessment,
# save_uploaded_file, extract_severity_level, display_severity_status, 
# create_comprehensive_pdf - they remain the same)

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'comprehensive_results' not in st.session_state:
        st.session_state.comprehensive_results = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None

    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MediScan Pro - Advanced Medical Analysis</h1>
        <p>Comprehensive Medical Scan Analysis & Health Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Patient Information Section
    st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            medical_history = st.text_area(
                "Previous Medical History",
                placeholder="e.g., Diabetes, Hypertension, Heart disease, Surgeries, etc.",
                height=100
            )
            current_medications = st.text_area(
                "Current Medications",
                placeholder="e.g., Metformin 500mg twice daily, Lisinopril 10mg once daily",
                height=100
            )
        
        with col2:
            lifestyle_diet = st.text_area(
                "Lifestyle & Diet",
                placeholder="e.g., Vegetarian, Regular exercise, Sedentary lifestyle, etc.",
                height=100
            )
            habits = st.text_area(
                "Habits",
                placeholder="e.g., Smoking, Alcohol consumption, Exercise routine, etc.",
                height=100
            )
    
    col3, col4 = st.columns(2)
    with col3:
        family_history = st.text_area(
            "Family Medical History",
            placeholder="e.g., Father - Diabetes, Mother - Heart disease, etc.",
            height=100
        )
    with col4:
        accidents = st.text_area(
            "Recent Accidents/Injuries",
            placeholder="e.g., Car accident last month, Fall injury, etc.",
            height=100
        )

    # Upload and Analysis Section
    st.markdown('<div class="section-header">📤 Upload Medical Scan</div>', unsafe_allow_html=True)
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload your medical scan report (X-ray, MRI, CT scan, Blood test, etc.)",
            type=["jpg", "jpeg", "png", "webp", "pdf"],
            help="Upload a clear image of your medical scan or report"
        )
        
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                st.info("📄 PDF uploaded successfully. Note: Image analysis works best with image files.")
            else:
                resized_image = resize_image_for_display(uploaded_file)
                if resized_image:
                    st.image(resized_image, caption="Uploaded Medical Scan", width=MAX_IMAGE_WIDTH)
                    file_size = len(uploaded_file.getvalue()) / 1024
                    st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB")
        
        # Analysis button
        if uploaded_file and st.button("🔬 Analyze Medical Scan", key="analyze_btn"):
            # Store patient data
            st.session_state.patient_data = {
                'age': age,
                'medical_history': medical_history,
                'medications': current_medications,
                'lifestyle': lifestyle_diet,
                'habits': habits,
                'family_history': family_history,
                'accidents': accidents
            }
            
            # Save and analyze file
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                try:
                    # Analyze medical scan
                    scan_analysis = analyze_medical_scan(temp_path)
                    
                    if scan_analysis:
                        st.session_state.analysis_results = scan_analysis
                        st.session_state.original_image = uploaded_file.getvalue()
                        
                        # Perform comprehensive assessment
                        comprehensive_analysis = comprehensive_health_assessment(
                            scan_analysis, 
                            st.session_state.patient_data
                        )
                        
                        if comprehensive_analysis:
                            st.session_state.comprehensive_results = comprehensive_analysis
                        
                        st.success("✅ Medical analysis completed successfully!")
                    else:
                        st.error("❌ Analysis failed. Please try with a clearer image.")
                
                except Exception as e:
                    st.error(f"🚨 Analysis failed: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

    # Results Section
    if st.session_state.analysis_results:
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        with st.container():
            # Extract and display severity
            severity = extract_severity_level(st.session_state.analysis_results)
            display_severity_status(severity)
            
            # Display scan analysis
            st.markdown("### 🔬 Medical Scan Analysis")
            st.markdown(st.session_state.analysis_results)
            
            # Display comprehensive assessment
            if st.session_state.comprehensive_results:
                st.markdown("---")
                st.markdown("### 📋 Comprehensive Health Assessment")
                st.markdown(st.session_state.comprehensive_results)
            
            # PDF Download
            if st.session_state.patient_data:
                st.markdown("---")
                st.subheader("📄 Download Complete Report")
                
                pdf_bytes = create_comprehensive_pdf(
                    st.session_state.original_image,
                    st.session_state.analysis_results,
                    st.session_state.comprehensive_results or "",
                    st.session_state.patient_data
                )
                
                if pdf_bytes:
                    download_filename = f"mediscan_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📥 Download Complete Medical Report",
                        data=pdf_bytes,
                        file_name=download_filename,
                        mime="application/pdf"
                    )

    # Emergency Information (only shown if needed)
    if st.session_state.analysis_results:
        severity = extract_severity_level(st.session_state.analysis_results)
        if severity in ["Emergency", "Severe"]:
            st.markdown("---")
            with st.container():
                st.markdown("### 🚨 Emergency Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **🏥 Immediate Actions:**
                    - Contact emergency services: 108/102
                    - Visit nearest emergency room
                    - Don't delay seeking medical attention
                    - Bring all medications and this report
                    """)
                with col2:
                    st.markdown("""
                    **👨‍⚕️ Specialists to Consult:**
                    - Emergency Medicine Physician
                    - Relevant specialist based on condition
                    - Primary care physician for follow-up
                    - Bring complete medical history
                    """)

    # Health Tips Section
    st.markdown("---")
    with st.container():
        st.markdown("### 💡 General Health Tips")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🍎 Nutrition**
            - Balanced diet with fruits & vegetables
            - Limit processed foods
            - Stay hydrated
            - Regular meal timing
            """)
        with col2:
            st.markdown("""
            **🏃‍♂️ Exercise**
            - Regular physical activity
            - 30 minutes daily walking
            - Strength training twice weekly
            - Consult doctor before starting
            """)
        with col3:
            st.markdown("""
            **😴 Lifestyle**
            - 7-8 hours quality sleep
            - Stress management
            - Regular health check-ups
            - Avoid smoking & excess alcohol
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #5a4a3a; font-size: 0.9rem; padding: 1rem;">
        © 2025 MediScan Pro - Advanced Medical Analysis Platform | 
        Powered by Gemini AI + Tavily | 
        🔒 Your health data is secure and private
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
