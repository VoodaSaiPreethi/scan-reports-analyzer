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
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Custom CSS for animated theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Animation */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Card Animations */
    .analysis-card {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        animation: slideInUp 0.8s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .input-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeInLeft 1s ease-out;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .results-section {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeInRight 1s ease-out;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Progress Bar Animation */
    .progress-container {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    /* Emergency Alert */
    .emergency-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: emergencyPulse 1.5s infinite;
        text-align: center;
        font-weight: 600;
    }
    
    .normal-status {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: normalGlow 2s infinite;
        text-align: center;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    /* Input Field Enhancements */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e3f2fd;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e3f2fd;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Animations */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        animation: slideInDown 0.8s ease-out;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes emergencyPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(255, 107, 107, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(255, 107, 107, 0);
        }
    }
    
    @keyframes normalGlow {
        0% {
            box-shadow: 0 0 5px rgba(0, 184, 148, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(0, 184, 148, 0.8);
        }
        100% {
            box-shadow: 0 0 5px rgba(0, 184, 148, 0.5);
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Medical Icons */
    .medical-icon {
        font-size: 2rem;
        margin-right: 1rem;
        vertical-align: middle;
    }
    
    /* Status Indicators */
    .status-normal {
        color: #00b894;
        font-weight: 600;
    }
    
    .status-warning {
        color: #fdcb6e;
        font-weight: 600;
    }
    
    .status-critical {
        color: #e17055;
        font-weight: 600;
    }
    
    .status-emergency {
        color: #d63031;
        font-weight: 600;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        50% {
            opacity: 0.5;
        }
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

MAX_IMAGE_WIDTH = 400

MEDICAL_ANALYSIS_PROMPT = """
You are a highly experienced medical AI assistant specializing in medical scan analysis and comprehensive health assessment.
Your role is to analyze medical scan reports (X-rays, MRIs, CT scans, blood tests, etc.) and provide detailed, accurate medical insights.

IMPORTANT INSTRUCTIONS:
1. Analyze the medical scan/report thoroughly
2. Identify specific medical conditions, abnormalities, or diseases
3. Provide detailed explanations in simple, layman language
4. Assess severity levels (Normal, Mild, Moderate, Severe, Emergency)
5. Consider patient's medical history, lifestyle, and medications
6. Provide specific precautions and recommendations
7. Determine if emergency consultation is needed
8. Suggest appropriate medical specialists to consult

ANALYSIS STRUCTURE:
- Primary Diagnosis: [Specific condition/disease name]
- Severity Level: [Normal/Mild/Moderate/Severe/Emergency]
- Detailed Explanation: [Simple language explanation]
- Probable Causes: [Based on scan and patient history]
- Symptoms to Watch: [What patient should monitor]
- Immediate Actions: [What to do right now]
- Precautions: [Lifestyle changes, things to avoid]
- Follow-up Required: [When and what type]
- Specialist Consultation: [Which doctor to see]
- Emergency Indicators: [When to seek immediate help]

Always prioritize patient safety and provide accurate, evidence-based medical information.
"""

PATIENT_HISTORY_PROMPT = """
You are a medical AI assistant specializing in comprehensive patient assessment.
Analyze the patient's complete medical profile including:
- Medical scan results
- Personal medical history
- Current medications
- Lifestyle factors
- Family medical history
- Recent accidents or injuries

Provide a holistic health assessment considering all factors and their interactions.
Focus on identifying risk factors, potential complications, and personalized recommendations.
"""

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=MEDICAL_ANALYSIS_PROMPT,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"❌ Error initializing medical agent: {e}")
        return None

@st.cache_resource
def get_comprehensive_agent():
    """Initialize and cache the comprehensive analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=PATIENT_HISTORY_PROMPT,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"❌ Error initializing comprehensive agent: {e}")
        return None

def resize_image_for_display(image_file):
    """Resize image for display only, returns bytes."""
    try:
        image_file.seek(0)
        img = Image.open(image_file)
        image_file.seek(0)
        
        aspect_ratio = img.height / img.width
        new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
        img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        st.error(f"🖼️ Error resizing image: {e}")
        return None

def analyze_medical_scan(image_path):
    """Analyze medical scan using AI."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing medical scan..."):
            response = agent.run(
                """Analyze this medical scan report thoroughly. Identify specific medical conditions, 
                assess severity, explain in simple terms, and provide detailed recommendations including 
                emergency indicators and specialist consultation needs.""",
                images=[image_path],
            )
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error analyzing medical scan: {e}")
        return None

def comprehensive_health_assessment(scan_analysis, patient_data):
    """Perform comprehensive health assessment."""
    agent = get_comprehensive_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🏥 Performing comprehensive health assessment..."):
            query = f"""
            Perform a comprehensive health assessment based on:
            
            MEDICAL SCAN ANALYSIS:
            {scan_analysis}
            
            PATIENT INFORMATION:
            Age: {patient_data['age']}
            Medical History: {patient_data['medical_history']}
            Current Medications: {patient_data['medications']}
            Lifestyle & Diet: {patient_data['lifestyle']}
            Habits: {patient_data['habits']}
            Family Medical History: {patient_data['family_history']}
            Recent Accidents/Injuries: {patient_data['accidents']}
            
            Provide a comprehensive assessment considering all factors, risk analysis, 
            personalized recommendations, and emergency indicators.
            """
            
            response = agent.run(query)
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error in comprehensive assessment: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk."""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        with NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        st.error(f"💾 Error saving uploaded file: {e}")
        return None

def extract_severity_level(analysis_text):
    """Extract severity level from analysis."""
    severity_patterns = {
        "Emergency": ["emergency", "critical", "urgent", "immediate"],
        "Severe": ["severe", "serious", "significant"],
        "Moderate": ["moderate", "moderately"],
        "Mild": ["mild", "minor", "slight"],
        "Normal": ["normal", "healthy", "no abnormalities"]
    }
    
    analysis_lower = analysis_text.lower()
    
    for severity, keywords in severity_patterns.items():
        if any(keyword in analysis_lower for keyword in keywords):
            return severity
    
    return "Unknown"

def display_severity_status(severity):
    """Display severity status with appropriate styling."""
    if severity == "Emergency":
        st.markdown('<div class="emergency-alert">🚨 EMERGENCY - SEEK IMMEDIATE MEDICAL ATTENTION</div>', unsafe_allow_html=True)
    elif severity == "Severe":
        st.error("🔴 SEVERE - Urgent medical consultation required")
    elif severity == "Moderate":
        st.warning("🟡 MODERATE - Medical consultation recommended")
    elif severity == "Mild":
        st.info("🟠 MILD - Monitor condition, routine check-up advised")
    elif severity == "Normal":
        st.markdown('<div class="normal-status">✅ NORMAL - No immediate concerns detected</div>', unsafe_allow_html=True)
    else:
        st.info("📊 Analysis completed - Review detailed results below")

def create_comprehensive_pdf(image_data, scan_analysis, comprehensive_analysis, patient_data):
    """Create a comprehensive PDF report."""
    try:
        buffer = BytesIO()
        pdf = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        content = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=20,
            alignment=1,
            spaceAfter=12,
            textColor=colors.navy
        )
        
        content.append(Paragraph("🏥 MediScan Pro - Comprehensive Medical Analysis Report", title_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Patient Information
        content.append(Paragraph("👤 Patient Information", styles['Heading2']))
        patient_info = f"""
        Age: {patient_data['age']}
        Medical History: {patient_data['medical_history']}
        Current Medications: {patient_data['medications']}
        Lifestyle & Diet: {patient_data['lifestyle']}
        """
        content.append(Paragraph(patient_info, styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Scan Analysis
        content.append(Paragraph("🔬 Medical Scan Analysis", styles['Heading2']))
        content.append(Paragraph(scan_analysis, styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Comprehensive Assessment
        content.append(Paragraph("📋 Comprehensive Health Assessment", styles['Heading2']))
        content.append(Paragraph(comprehensive_analysis, styles['Normal']))
        
        # Date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Spacer(1, 0.3*inch))
        content.append(Paragraph(f"Generated on: {current_datetime}", styles['Normal']))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"📄 Error creating PDF: {e}")
        return None

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

    # Sidebar for patient information
    with st.sidebar:
        st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)
        
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
        
        family_history = st.text_area(
            "Family Medical History",
            placeholder="e.g., Father - Diabetes, Mother - Heart disease, etc.",
            height=100
        )
        
        accidents = st.text_area(
            "Recent Accidents/Injuries",
            placeholder="e.g., Car accident last month, Fall injury, etc.",
            height=100
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📤 Upload Medical Scan</div>', unsafe_allow_html=True)
        
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
    
    with col2:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
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
        else:
            st.info("👆 Upload a medical scan and complete patient information to see detailed analysis results here.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Emergency Information Section
    if st.session_state.analysis_results:
        severity = extract_severity_level(st.session_state.analysis_results)
        if severity in ["Emergency", "Severe"]:
            st.markdown("---")
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
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
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Health Tips Section
    st.markdown("---")
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        © 2025 MediScan Pro - Advanced Medical Analysis Platform | 
        Powered by Gemini AI + Tavily | 
        🔒 Your health data is secure and private
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
