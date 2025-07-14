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
    page_title="MediScan - Medical Report Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏥"
)

# Custom CSS for nude/light theme
st.markdown("""
<style>
    .main {
        background-color: #FAF9F6;
        padding: 2rem;
    }
    
    .stApp {
        background-color: #F5F5DC;
    }
    
    .stTitle {
        color: #8B4513;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 2rem;
    }
    
    .stSubheader {
        color: #A0522D;
        font-family: 'Arial', sans-serif;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: #FFF8DC;
        border: 1px solid #D2B48C;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #FFF8DC;
        border: 1px solid #D2B48C;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stSelectbox > div > div > select {
        background-color: #FFF8DC;
        border: 1px solid #D2B48C;
        border-radius: 8px;
    }
    
    .stButton > button {
        background-color: #DEB887;
        color: #8B4513;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #D2B48C;
    }
    
    .info-card {
        background-color: #FFF8DC;
        border: 1px solid #D2B48C;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .emergency-alert {
        background-color: #FFE4E1;
        border: 2px solid #CD5C5C;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .section-divider {
        border-bottom: 2px solid #D2B48C;
        margin: 2rem 0;
    }
    
    .linear-layout {
        display: flex;
        flex-direction: column;
        gap: 1rem;
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

       # Update the MEDICAL_SCAN_SYSTEM_PROMPT to emphasize condition naming
MEDICAL_SCAN_SYSTEM_PROMPT = """
You are an expert medical diagnostician and radiologist with extensive knowledge in medical imaging interpretation and patient care.
Your role is to analyze medical scan reports (X-rays, CT scans, MRIs, ultrasounds, blood tests, etc.) and provide comprehensive medical insights.

You must:
1. Clearly identify and NAME all medical conditions found in the scan using precise medical terminology
2. Provide layman-friendly explanations of each condition
3. List conditions in order of clinical significance
4. For each condition, specify:
   - Exact medical name
   - Location in the body
   - Size/extent if applicable
   - Any measurable parameters
5. Assess severity and urgency of findings
6. Provide detailed explanations of what the conditions mean for the patient's health
7. Suggest appropriate precautions and lifestyle modifications
8. Determine if the condition requires immediate medical attention
9. Consider the patient's complete medical history in your analysis
10. Provide specific recommendations for specialist consultations when needed

Always use precise medical terminology first, followed by simple explanations.
"""

# Update the analysis instructions for clearer structure
MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan report and provide a comprehensive analysis in the following structured format:

*Diagnosed Conditions:* 
<Bulleted list of ALL medical conditions identified with exact medical names>
For each condition:
- Exact medical name (primary diagnosis)
- Location in body
- Size/extent if applicable
- Clinical significance

*Detailed Analysis:* <detailed analysis of what the scan shows>
*Condition Explanations:* <explain each condition in simple, understandable language>
*Severity Assessment:* <rate severity: Low/Moderate/High/Critical with justification>
*Immediate Concerns:* <any urgent issues that need immediate attention>
*Recommended Actions:* <specific medical actions to take>
*Precautions:* <specific precautions to take>
*Lifestyle Modifications:* <diet, exercise, habits to change>
*Specialist Referrals:* <which doctors to consult and timeline>
*Emergency Indicators:* <symptoms that require immediate medical help>
*Follow-up Plan:* <when to get re-tested or follow-up scans>
*Prognosis:* <what to expect going forward>

Format conditions clearly and prioritize by clinical importance.
"""

# In the main function, update the results display section to highlight conditions first
if st.session_state.scan_analysis:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 Medical Analysis Results")
    
    # Parse and display diagnosed conditions first
    st.markdown("### 🩺 Diagnosed Conditions")
    conditions_match = re.search(r"\*Diagnosed Conditions:\*(.*?)(?=\*(?:Detailed Analysis|Condition Explanations):\*)", 
                               st.session_state.scan_analysis, re.DOTALL | re.IGNORECASE)
    
    if conditions_match:
        conditions_content = conditions_match.group(1).strip()
        # Add special styling for conditions
        st.markdown(f"""
        <div style="background-color:#FFF0F5; padding:15px; border-radius:10px; border-left:5px solid #FF69B4">
        {conditions_content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No specific conditions identified in the analysis")
    
    st.markdown("---")
    
    # Rest of the analysis sections...
    sections = [
        "Detailed Analysis", "Condition Explanations", "Severity Assessment", 
        "Immediate Concerns", "Recommended Actions", "Precautions",
        "Lifestyle Modifications", "Specialist Referrals", "Emergency Indicators",
        "Follow-up Plan", "Prognosis"
    ]
    
    for section in sections:
        pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections)}):\*|$)"
        match = re.search(pattern, st.session_state.scan_analysis, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            st.markdown(f"**{section.replace('_', ' ')}:**")
            
            if section == "Severity Assessment":
                # Add color coding for severity
                if "Critical" in content or "High" in content:
                    st.error(content)
                elif "Moderate" in content:
                    st.warning(content)
                else:
                    st.success(content)
            else:
                st.write(content)
            
            st.markdown("---") 
        # Display comprehensive analysis
        if st.session_state.comprehensive_analysis:
            st.markdown("### 📊 Comprehensive Health Assessment")
            st.write(st.session_state.comprehensive_analysis)
            st.markdown("---")
        
        # PDF download
        if st.session_state.original_image:
            st.subheader("📄 Download Medical Report")
            
            pdf_bytes = create_medical_pdf(
                st.session_state.original_image,
                st.session_state.scan_analysis,
                st.session_state.comprehensive_analysis or "Comprehensive analysis not available",
                st.session_state.patient_info
            )
            
            if pdf_bytes:
                download_filename = f"medical_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥 Download Complete Medical Report",
                    data=pdf_bytes,
                    file_name=download_filename,
                    mime="application/pdf",
                    help="Download a comprehensive PDF report with all analysis results and recommendations"
                )
    
    # Important Health Reminders
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🩺 Important Health Reminders")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🚨 When to Seek Emergency Care:**
        - Severe chest pain or difficulty breathing
        - Sudden severe headache or vision changes
        - Signs of stroke (facial drooping, arm weakness, speech difficulty)
        - Severe abdominal pain
        - Heavy bleeding or trauma
        """)
        
        st.markdown("""
        **👨‍⚕️ Regular Health Monitoring:**
        - Annual physical examinations
        - Age-appropriate screenings
        - Medication adherence and monitoring
        - Lifestyle modifications as recommended
        """)
    
    with col2:
        st.markdown("""
        **📞 Emergency Contacts:**
        - Emergency Services: 911 (US) / 112 (EU) / Local emergency number
        - Poison Control: Contact your local poison control center
        - Mental Health Crisis: National crisis helpline
        - Your Primary Care Physician
        """)
        
        st.markdown("""
        **💊 Medication Safety:**
        - Take medications exactly as prescribed
        - Never share medications with others
        - Report side effects to your healthcare provider
        - Keep an updated medication list
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("© 2025 MediScan Medical Report Analyzer | AI-Powered Medical Analysis | For Educational Purposes Only")

if __name__ == "__main__":
    main()
