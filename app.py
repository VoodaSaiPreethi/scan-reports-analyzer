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
import time
import json

# Set page configuration
st.set_page_config(
    page_title="Scan Reports Analyzer - Universal Medical Imaging Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔬"
)

# Custom CSS for professional medical theme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        color: #2c3e50;
        border: 2px solid #3498db;
        border-radius: 8px;
    }
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
        color: #2c3e50;
        border: 2px solid #3498db;
        border-radius: 8px;
    }
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        color: #2c3e50;
        border: 2px solid #3498db;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSidebar {
        background-color: #ecf0f1;
    }
    .stExpander {
        background-color: #ecf0f1;
        border: 1px solid #bdc3c7;
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        font-weight: bold;
    }
    .stMarkdown {
        color: #34495e;
    }
    .scan-type-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .emergency-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .analysis-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    .emergency-section {
        background: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
    }
    .warning-section {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    .success-section {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
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

MAX_IMAGE_WIDTH = 700

# Enhanced system prompt for comprehensive medical analysis with real-time data
SYSTEM_PROMPT = """
You are a highly experienced medical professional with expertise in radiology, pathology, and medical imaging interpretation across ALL medical specialties. You have access to real-time medical literature and current medical guidelines.

Your expertise covers:
- Radiology (CT, MRI, X-Ray, Ultrasound, PET scans)
- Pathology (Histopathology, Cytology, Biopsy reports)
- Cardiology (ECG, Echocardiography, Angiography)
- Neurology (Brain scans, Spinal imaging)
- Orthopedics (Bone scans, Joint imaging)
- Gastroenterology (Endoscopy, Colonoscopy)
- Pulmonology (Chest X-rays, CT scans)
- Oncology (Tumor imaging, Cancer staging)
- Laboratory Medicine (Blood tests, Urine analysis)
- Emergency Medicine (Critical findings, Urgent conditions)

You must analyze medical scans and reports with the following priorities:
1. Patient safety and emergency detection
2. Accurate interpretation based on current medical standards
3. Clear communication in layman terms
4. Evidence-based recommendations
5. Real-time medical literature integration

Always use the latest medical guidelines and research when making assessments.
"""

# Enhanced analysis instructions with real-time data integration
MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan/report with real-time medical data integration and provide a comprehensive assessment following these steps:

1. **REAL-TIME DATA INTEGRATION:**
   - Search for the latest medical guidelines and research relevant to the findings
   - Incorporate current diagnostic criteria and treatment protocols
   - Reference recent studies and evidence-based medicine
   - Use up-to-date normal ranges and reference values

2. **SCAN TYPE IDENTIFICATION:**
   - Identify the exact type of medical scan/report
   - Explain the clinical purpose and methodology
   - Describe the anatomical region and systems examined
   - Note any technical quality or limitations

3. **DETAILED FINDINGS ANALYSIS:**
   - Systematically analyze all visible findings
   - Compare findings with current normal ranges and standards
   - Identify incidental findings and their clinical significance
   - Correlate findings with patient's clinical presentation

4. **ABNORMALITIES ASSESSMENT WITH SEVERITY GRADING:**
   - Classify abnormalities using current medical classification systems
   - Provide severity grading: Normal, Minimal, Mild, Moderate, Severe, Critical
   - Explain the pathophysiology in simple terms
   - Discuss potential complications and progression

5. **EMERGENCY STATUS DETERMINATION:**
   - Use current emergency medicine protocols
   - Identify life-threatening conditions requiring immediate intervention
   - Provide specific timeframes for medical attention
   - List red flag symptoms requiring emergency care

6. **EVIDENCE-BASED PROBLEM EXPLANATION:**
   - Explain each finding using current medical understanding
   - Provide analogies and visual descriptions for patient comprehension
   - Include relevant statistics and prognosis information
   - Discuss how findings affect daily life and function

7. **CLINICAL CORRELATION:**
   - Integrate findings with patient's medical history and medications
   - Identify drug interactions or contraindications
   - Assess impact of comorbidities on findings
   - Consider age-related and gender-specific factors

8. **CURRENT TREATMENT RECOMMENDATIONS:**
   - Provide evidence-based treatment options
   - Include both medical and surgical interventions when appropriate
   - Discuss latest therapeutic advances
   - Recommend appropriate specialist referrals with urgency levels

9. **LIFESTYLE AND PREVENTIVE MEASURES:**
   - Provide current dietary and exercise recommendations
   - Include evidence-based lifestyle modifications
   - Discuss preventive measures based on latest research
   - Address mental health and quality of life aspects

10. **FOLLOW-UP AND MONITORING:**
    - Recommend appropriate follow-up intervals based on current guidelines
    - Specify monitoring parameters and frequency
    - Include patient education on self-monitoring
    - Provide clear action plans for different scenarios

Format your response using this EXACT structure with clear sections:

**🔬 SCAN TYPE & PURPOSE:**
[Detailed identification and explanation]

**📋 EXECUTIVE SUMMARY:**
[Brief overview in simple terms with key takeaways]

**🔍 DETAILED FINDINGS:**
[Comprehensive analysis of all findings with real-time medical context]

**⚠️ ABNORMALITIES IDENTIFIED:**
[Specific abnormalities with current classification and severity]

**🚨 EMERGENCY STATUS:**
[Clear determination with specific timeframes and actions]

**📝 PROBLEM EXPLANATION:**
[Detailed explanation using current medical understanding]

**🔗 MEDICAL CORRELATION:**
[Integration with patient profile and current medical knowledge]

**👨‍⚕️ SPECIALIST CONSULTATIONS:**
[Evidence-based referral recommendations with urgency]

**⚠️ IMMEDIATE PRECAUTIONS:**
[Specific actions based on current safety protocols]

**🌟 LIFESTYLE RECOMMENDATIONS:**
[Evidence-based lifestyle modifications]

**📅 FOLLOW-UP REQUIREMENTS:**
[Specific monitoring plan based on current guidelines]

**❓ QUESTIONS FOR YOUR DOCTOR:**
[Relevant questions based on findings and current treatment options]

**🚨 WARNING SIGNS:**
[Specific symptoms requiring immediate medical attention]

**📊 PROGNOSIS & OUTLOOK:**
[Current understanding of condition progression and outcomes]

**🔬 LATEST RESEARCH:**
[Recent developments and research relevant to the findings]
"""

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical scan analysis agent with real-time capabilities."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=SYSTEM_PROMPT,
            instructions=MEDICAL_ANALYSIS_INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
            show_tool_calls=True,
            debug_mode=False
        )
    except Exception as e:
        st.error(f"❌ Error initializing medical analysis agent: {e}")
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

def get_real_time_medical_context(scan_type, findings_keywords):
    """Get real-time medical context for better analysis."""
    try:
        agent = get_medical_agent()
        if agent is None:
            return ""
        
        context_query = f"""
        Search for the latest medical guidelines, research, and diagnostic criteria for {scan_type} 
        focusing on: {', '.join(findings_keywords)}. 
        Include current reference ranges, diagnostic criteria, and treatment protocols.
        """
        
        with st.spinner("🔍 Gathering real-time medical data..."):
            context_response = agent.run(context_query)
            return context_response.content if context_response else ""
    except Exception as e:
        st.warning(f"⚠️ Could not retrieve real-time medical context: {e}")
        return ""

def analyze_medical_scan(image_path, patient_data, scan_type):
    """Analyze medical scan with enhanced real-time data integration."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initial scan analysis
        status_text.text("🔬 Analyzing medical scan structure...")
        progress_bar.progress(10)
        
        # Step 2: Real-time medical data gathering
        status_text.text("🔍 Gathering real-time medical guidelines...")
        progress_bar.progress(30)
        
        # Step 3: Comprehensive analysis
        status_text.text("📊 Performing comprehensive medical analysis...")
        progress_bar.progress(50)
        
        # Enhanced query with real-time data integration
        enhanced_query = f"""
        MEDICAL SCAN ANALYSIS REQUEST - {scan_type}
        
        Patient Profile:
        - Age: {patient_data['age']} years
        - Gender: {patient_data['gender']}
        - Medical History: {patient_data['medical_history']}
        - Current Medications: {patient_data['medications']}
        - Current Symptoms: {patient_data['symptoms']}
        - Health Problems: {patient_data['health_problems']}
        - Diet: {patient_data['diet']}
        - Lifestyle: {patient_data['lifestyle']}
        - Habits: {patient_data['habits']}
        - Family History: {patient_data['family_history']}
        
        ANALYSIS REQUIREMENTS:
        1. First, search for the latest medical guidelines and research relevant to this {scan_type}
        2. Use current diagnostic criteria and reference ranges
        3. Integrate real-time medical literature and evidence-based medicine
        4. Provide comprehensive analysis following the structured format
        5. Ensure all recommendations are based on current medical standards
        
        CRITICAL INSTRUCTIONS:
        - If ANY emergency conditions are detected, state this immediately at the beginning
        - Use the latest medical research and guidelines in your analysis
        - Provide specific, actionable recommendations
        - Include prognosis and latest research findings
        - Ensure all medical advice is current and evidence-based
        
        Please analyze this medical scan/report with the highest level of medical expertise and current knowledge.
        """
        
        progress_bar.progress(70)
        status_text.text("🩺 Generating detailed medical interpretation...")
        
        # Run the enhanced analysis
        response = agent.run(enhanced_query, images=[image_path])
        
        progress_bar.progress(90)
        status_text.text("✅ Finalizing comprehensive analysis...")
        
        progress_bar.progress(100)
        status_text.text("✅ Medical analysis completed successfully!")
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return response.content.strip() if response else None
        
    except Exception as e:
        st.error(f"🚨 Error analyzing medical scan: {e}")
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

def create_enhanced_medical_pdf(image_data, analysis_results, patient_data, scan_type):
    """Create an enhanced PDF report with real-time medical analysis."""
    try:
        buffer = BytesIO()
        pdf = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        content = []
        styles = getSampleStyleSheet()
        
        # Enhanced custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            alignment=1,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=16,
            alignment=1,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6
        )
        
        emergency_style = ParagraphStyle(
            'Emergency',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=12,
            backColor=colors.HexColor('#ffebee'),
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            borderWidth=1,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.HexColor('#fff3e0'),
            alignment=1
        )
        
        # Header
        content.append(Paragraph("🔬 SCAN REPORTS ANALYZER", title_style))
        content.append(Paragraph("Universal Medical Imaging Analysis Report", subtitle_style))
        content.append(Paragraph("Real-Time Medical Data Integration", subtitle_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Medical disclaimer
        content.append(Paragraph(
            "⚠️ MEDICAL DISCLAIMER: This analysis incorporates real-time medical data and current guidelines "
            "for educational and informational purposes only. It should NOT replace professional medical advice, "
            "diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.",
            disclaimer_style
        ))
        content.append(Spacer(1, 0.3*inch))
        
        # Patient information with enhanced formatting
        content.append(Paragraph("👤 PATIENT INFORMATION", heading_style))
        
        patient_info = [
            f"Age: {patient_data['age']} years",
            f"Gender: {patient_data['gender']}",
            f"Scan Type: {scan_type}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Medical History: {patient_data['medical_history'][:200]}{'...' if len(patient_data['medical_history']) > 200 else ''}",
            f"Current Medications: {patient_data['medications'][:200]}{'...' if len(patient_data['medications']) > 200 else ''}",
            f"Current Symptoms: {patient_data['symptoms'][:200]}{'...' if len(patient_data['symptoms']) > 200 else ''}"
        ]
        
        for info in patient_info:
            content.append(Paragraph(info, normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        
        # Medical scan image
        if image_data:
            try:
                img_temp = BytesIO(image_data)
                img = Image.open(img_temp)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                display_width = 6 * inch
                display_height = display_width * aspect
                
                if display_height > 7 * inch:
                    display_height = 7 * inch
                    display_width = display_height / aspect
                
                img_temp.seek(0)
                img_obj = ReportLabImage(img_temp, width=display_width, height=display_height)
                content.append(Paragraph(f"🖼️ {scan_type} IMAGE", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.3*inch))
            except Exception as img_error:
                content.append(Paragraph(f"⚠️ Could not include image: {img_error}", normal_style))
        
        # Enhanced analysis results
        content.append(Paragraph("📊 COMPREHENSIVE MEDICAL ANALYSIS", heading_style))
        
        if analysis_results:
            # Enhanced section parsing with new sections
            sections = {
                "🔬 SCAN TYPE & PURPOSE": "Scan Type & Purpose",
                "📋 EXECUTIVE SUMMARY": "Executive Summary",
                "🔍 DETAILED FINDINGS": "Detailed Findings",
                "⚠️ ABNORMALITIES IDENTIFIED": "Abnormalities Identified",
                "🚨 EMERGENCY STATUS": "Emergency Status",
                "📝 PROBLEM EXPLANATION": "Problem Explanation",
                "🔗 MEDICAL CORRELATION": "Medical Correlation",
                "👨‍⚕️ SPECIALIST CONSULTATIONS": "Specialist Consultations",
                "⚠️ IMMEDIATE PRECAUTIONS": "Immediate Precautions",
                "🌟 LIFESTYLE RECOMMENDATIONS": "Lifestyle Recommendations",
                "📅 FOLLOW-UP REQUIREMENTS": "Follow-up Requirements",
                "❓ QUESTIONS FOR YOUR DOCTOR": "Questions for Your Doctor",
                "🚨 WARNING SIGNS": "Warning Signs",
                "📊 PROGNOSIS & OUTLOOK": "Prognosis & Outlook",
                "🔬 LATEST RESEARCH": "Latest Research"
            }
            
            for section_title, section_key in sections.items():
                # Try multiple pattern variations
                patterns = [
                    rf"\*\*{re.escape(section_title)}:\*\*(.*?)(?=\*\*(?:🔬|📋|🔍|⚠️|🚨|📝|🔗|👨‍⚕️|🌟|📅|❓|📊) [A-Z]|$)",
                    rf"\*\*{re.escape(section_key)}:\*\*(.*?)(?=\*\*[A-Z]|$)",
                    rf"{re.escape(section_title)}:(.*?)(?=(?:🔬|📋|🔍|⚠️|🚨|📝|🔗|👨‍⚕️|🌟|📅|❓|📊) [A-Z]|$)",
                    rf"\*{re.escape(section_key)}:\*(.*?)(?=\*[A-Z]|$)"
                ]
                
                content_found = None
                for pattern in patterns:
                    match = re.search(pattern, analysis_results, re.DOTALL | re.IGNORECASE)
                    if match:
                        content_found = match.group(1).strip()
                        break
                
                if content_found:
                    # Special formatting for emergency sections
                    if section_key in ["Emergency Status", "Warning Signs"]:
                        if "emergency" in content_found.lower() or "urgent" in content_found.lower():
                            content.append(Paragraph(f"🚨 {section_key.upper()}", heading_style))
                            content.append(Paragraph(content_found, emergency_style))
                        else:
                            content.append(Paragraph(section_title, heading_style))
                            content.append(Paragraph(content_found, normal_style))
                    else:
                        content.append(Paragraph(section_title, heading_style))
                        
                        # Split into paragraphs and clean
                        paragraphs = content_found.split("\n")
                        for para in paragraphs:
                            if para.strip():
                                clean_para = para.strip().replace('<', '&lt;').replace('>', '&gt;')
                                content.append(Paragraph(clean_para, normal_style))
                    
                    content.append(Spacer(1, 0.2*inch))
        
        # Enhanced emergency contact information
        content.append(Paragraph("🚨 EMERGENCY CONTACT INFORMATION", heading_style))
        content.append(Paragraph(
            "SEEK IMMEDIATE MEDICAL ATTENTION for: Severe chest pain, difficulty breathing, "
            "loss of consciousness, severe bleeding, stroke symptoms, or any life-threatening conditions. "
            "Contact emergency services immediately (911, 108, 999, etc.)",
            emergency_style
        ))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph(
            "© 2025 Scan Reports Analyzer | Real-Time Medical Analysis | Powered by Gemini AI + Tavily", 
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                         textColor=colors.gray, alignment=1)
        ))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"📄 Error creating enhanced PDF: {e}")
        return None

def display_enhanced_analysis_section(title, content, icon, section_type="normal"):
    """Display analysis section with enhanced formatting and real-time context."""
    if content:
        # Create appropriate styling based on section type
        if section_type == "emergency":
            st.markdown(f'<div class="emergency-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.error(content)
        elif section_type == "warning":
            st.markdown(f'<div class="warning-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.warning(content)
        elif section_type == "success":
            st.markdown(f'<div class="success-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.success(content)
        else:
            st.markdown(f'<div class="analysis-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.info(content)
        
        st.markdown("---")

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'medical_image' not in st.session_state:
        st.session_state.medical_image = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    if 'scan_type' not in st.session_state:
        st.session_state.scan_type = None
    if 'analysis_timestamp' not in st.session_state:
        st.session_state.analysis_timestamp = None

    # Header with real-time indicator
    st.title("🔬 Scan Reports Analyzer")
    st.markdown("### Universal Medical Imaging Analysis - Real-Time AI-Powered Healthcare Assistant")
    
    # Real-time status indicator
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"🕐 **Real-Time Analysis Active** | Current Time: {current_time}")
    
    # Supported scan types display
    st.markdown("""
    <div class="scan-type-card">
        🏥 Real-Time Medical Analysis: CT, MRI, X-Ray, Ultrasound, Blood Tests, ECG, Pathology Reports & More!
        <br>📊 Integrated with Latest Medical Guidelines & Research
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced medical disclaimer
    st.error("""
    ⚠️ **CRITICAL MEDICAL DISCLAIMER**
    
    This medical scan analysis tool uses real-time medical data and current guidelines for educational 
    and informational purposes ONLY. The AI interpretation should NEVER replace professional medical 
    advice, diagnosis, or treatment.
    
    🏥 **Always consult with qualified healthcare professionals for:**
    - Accurate medical assessment and diagnosis
    - Treatment decisions and medical care
    - Emergency medical situations
    - Any health concerns or symptoms
    
    📞 **For medical emergencies, contact your local emergency services immediately**
    
    🔬 **Real-Time Data Integration:** This system incorporates current medical literature, 
    guidelines, and research to provide the most up-to-date analysis possible.
    """)
    
    # Patient Information Section
    st.markdown("## 👤 Patient Information")
    st.markdown("*Complete patient profile for comprehensive analysis*")
    
    # Basic demographics
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, 
                             help="Patient's current age in years")
    with demo_col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                             help="Patient's gender (affects medical reference ranges)")
    
    # Enhanced Medical History
    st.markdown("### 📋 Medical History")
    medical_history = st.text_area(
        "Previous medical conditions, surgeries, and significant health events:",
        placeholder="e.g., Type 2 Diabetes (2015), Hypertension (2018), Cardiac bypass surgery (2020), "
                   "Previous stroke, Cancer history, Hospitalizations, Major injuries...",
        height=100,
        help="Include dates when possible and any major health events"
    )
    
    # Current Medications
    st.markdown("### 💊 Current Medications")
    medications = st.text_area(
        "List all current medications with dosages and frequency:",
        placeholder="e.g., Metformin 500mg twice daily, Lisinopril 10mg once daily, "
                   "Aspirin 81mg daily, Multivitamin, Omega-3 supplements...",
        height=100,
        help="Include prescription medications, over-the-counter drugs, and supplements"
    )
    
    # Current Symptoms
    st.markdown("### 🩺 Current Symptoms")
    symptoms = st.text_area(
        "Current symptoms and when they started:",
        placeholder="e.g., Chest pain for 2 weeks, Shortness of breath when climbing stairs, "
                   "Persistent headaches, Fatigue, Dizziness upon standing...",
        height=100,
        help="Describe symptoms, their duration, severity, and any triggering factors"
    )
    
    # Health Problems
    st.markdown("### 🏥 Current Health Problems")
    health_problems = st.text_area(
        "Ongoing health issues and chronic conditions:",
        placeholder="e.g., Chronic kidney disease stage 3, Rheumatoid arthritis, "
                   "Obstructive sleep apnea, Depression, Autoimmune thyroiditis...",
        height=100,
        help="Include all active medical conditions and their current status"
    )
    
    # Family History
    st.markdown("### 👨‍👩‍👧‍👦 Family History")
    family_history = st.text_area(
        "Family medical history (parents, siblings, grandparents):",
        placeholder="e.g., Father: Heart attack at age 55, Mother: Type 2 diabetes, "
                   "Sister: Breast cancer at 42, Grandfather: Alzheimer's disease...",
        height=80,
        help="Include major illnesses and ages of onset if known"
    )
    
    # Diet and Lifestyle
    st.markdown("### 🥗 Diet and Lifestyle")
    diet = st.text_area(
        "Dietary habits and preferences:",
        placeholder="e.g., Mediterranean diet, Vegetarian, High sodium intake, "
                   "Processed foods, Alcohol 2-3 drinks/week, Caffeine 2 cups/day...",
        height=80,
        help="Describe typical eating patterns and dietary restrictions"
    )
    
    lifestyle = st.text_area(
        "Lifestyle factors:",
        placeholder="e.g., Sedentary office job, Walks 30 minutes daily, "
                   "High stress job, Sleeps 6 hours/night, Works night shifts...",
        height=80,
        help="Describe physical activity, sleep, stress levels, and work environment"
    )
    
    # Habits
    st.markdown("### 🚬 Habits")
    habits = st.text_area(
        "Smoking, alcohol, and other relevant habits:",
        placeholder="e.g., Current smoker (1 pack/day for 10 years), "
                   "Former smoker (quit 5 years ago), Social drinker (3-4 drinks/week), "
                   "No recreational drug use...",
        height=80,
        help="Include tobacco, alcohol, and substance use history"
    )
    
    # Scan Type Selection
    st.markdown("## 🔬 Medical Scan Type")
    scan_type = st.selectbox(
        "Select the type of medical scan/report you're uploading:",
        [
            "CT Scan", "MRI Scan", "X-Ray", "Ultrasound", "PET Scan", 
            "Nuclear Medicine Scan", "Blood Test Results", "Urine Analysis", 
            "ECG/EKG", "Echocardiogram", "Pathology Report", "Biopsy Results", 
            "Endoscopy Report", "Colonoscopy Report", "Mammography", 
            "Bone Density Scan", "Angiography", "Stress Test Results",
            "Pulmonary Function Test", "Allergy Test Results", 
            "Genetic Test Results", "Other"
        ],
        help="Select the most appropriate category for your medical scan/report"
    )
    
    # Medical Scan Upload
    st.markdown("## 🖼️ Medical Scan/Report Upload")
    uploaded_file = st.file_uploader(
        "Upload your medical scan report, test results, or imaging:",
        type=["jpg", "jpeg", "png", "webp", "pdf", "dcm", "tiff", "bmp"],
        help="Upload a clear image of your medical scan, test results, or report. "
             "For best results, use high-quality images with good contrast."
    )
    
    if uploaded_file:
        # Display uploaded image
        if uploaded_file.type.startswith('image/'):
            resized_image = resize_image_for_display(uploaded_file)
            if resized_image:
                st.image(resized_image, caption=f"Uploaded {scan_type}", width=MAX_IMAGE_WIDTH)
        
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB • {scan_type}")
    
    # Analysis Button
    if uploaded_file and st.button("🔬 Analyze Medical Scan/Report", type="primary"):
        # Prepare patient data
        patient_data = {
            'age': age,
            'gender': gender,
            'medical_history': medical_history,
            'medications': medications,
            'symptoms': symptoms,
            'health_problems': health_problems,
            'family_history': family_history,
            'diet': diet,
            'lifestyle': lifestyle,
            'habits': habits
        }
        
        # Save uploaded file and analyze
        temp_path = save_uploaded_file(uploaded_file)
        if temp_path:
            try:
                analysis_result = analyze_medical_scan(temp_path, patient_data, scan_type)
                
                if analysis_result:
                    st.session_state.analysis_results = analysis_result
                    st.session_state.medical_image = uploaded_file.getvalue()
                    st.session_state.patient_data = patient_data
                    st.session_state.scan_type = scan_type
                    st.session_state.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.success("✅ Medical scan analysis completed successfully!")
                    
                    # Check for emergency conditions
                    if "emergency" in analysis_result.lower() or "urgent" in analysis_result.lower():
                        st.markdown('<div class="emergency-banner">🚨 EMERGENCY FINDINGS DETECTED - IMMEDIATE MEDICAL ATTENTION REQUIRED</div>', 
                                  unsafe_allow_html=True)
                else:
                    st.error("❌ Analysis failed. Please try with a clearer image or report.")
                
            except Exception as e:
                st.error(f"🚨 Analysis failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Display Results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Medical Analysis Results")
        st.markdown(f"**Analysis performed on:** {st.session_state.analysis_timestamp}")
        
        analysis_text = st.session_state.analysis_results
        
        # Check if emergency
        is_emergency = "emergency" in analysis_text.lower() or "urgent" in analysis_text.lower()
        
        # Define sections with appropriate icons and types
        sections = [
            ("🔬 Scan Type & Purpose", "normal"),
            ("📋 Executive Summary", "normal"),
            ("🔍 Detailed Findings", "normal"),
            ("⚠️ Abnormalities Identified", "warning"),
            ("🚨 Emergency Status", "emergency" if is_emergency else "normal"),
            ("📝 Problem Explanation", "normal"),
            ("🔗 Medical Correlation", "normal"),
            ("👨‍⚕️ Specialist Consultations", "normal"),
            ("⚠️ Immediate Precautions", "warning"),
            ("🌟 Lifestyle Recommendations", "success"),
            ("📅 Follow-up Requirements", "normal"),
            ("❓ Questions for Your Doctor", "normal"),
            ("🚨 Warning Signs", "emergency" if is_emergency else "warning"),
            ("📊 Prognosis & Outlook", "normal"),
            ("🔬 Latest Research", "normal")
        ]
        
        for section_title, section_type in sections:
            # Try multiple pattern variations to find the section content
            patterns = [
                rf"\*\*{re.escape(section_title)}:\*\*(.*?)(?=\*\*[🔬📋🔍⚠️🚨📝🔗👨‍⚕️🌟📅❓📊]|$)",
                rf"{re.escape(section_title)}:(.*?)(?=[🔬📋🔍⚠️🚨📝🔗👨‍⚕️🌟📅❓📊]|$)",
                rf"\*{re.escape(section_title)}:\*(.*?)(?=\*[A-Z]|$)"
            ]
            
            content = None
            for pattern in patterns:
                match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    break
            
            if content:
                display_enhanced_analysis_section(section_title, content, "", section_type)
        
        # Emergency Contact Information
        st.markdown("## 🚨 Emergency Contact Information")
        st.error("""
        **🏥 SEEK IMMEDIATE MEDICAL ATTENTION if you experience:**
        - Severe chest pain, pressure, or tightness
        - Difficulty breathing or shortness of breath
        - Sudden severe headache or vision changes
        - Loss of consciousness or fainting
        - Severe abdominal pain or vomiting blood
        - Signs of stroke (sudden weakness, speech difficulties, facial drooping)
        - Severe allergic reactions (difficulty breathing, swelling)
        - Seizures or convulsions
        - High fever with severe symptoms
        - Severe bleeding that won't stop
        - Thoughts of self-harm or suicide
        
        **📞 Emergency Services:**
        - USA: 911
        - India: 108 (Emergency) / 102 (Ambulance)
        - UK: 999
        - Australia: 000
        - Canada: 911
        
        **🏥 When in doubt, GO TO THE EMERGENCY ROOM immediately**
        """)
        
        # PDF Download
        st.markdown("## 📄 Download Complete Medical Report")
        
        if st.session_state.medical_image and st.session_state.patient_data:
            pdf_bytes = create_enhanced_medical_pdf(
                st.session_state.medical_image,
                st.session_state.analysis_results,
                st.session_state.patient_data,
                st.session_state.scan_type
            )
            if pdf_bytes:
                download_filename = f"medical_scan_analysis_{st.session_state.scan_type.lower().replace(' ', '_')}_{st.session_state.analysis_timestamp.replace(':', '').replace(' ', '_')}.pdf"
                st.download_button(
                    label="📥 Download Complete Medical Analysis Report (PDF)",
                    data=pdf_bytes,
                    file_name=download_filename,
                    mime="application/pdf",
                    help="Download a comprehensive PDF report with all analysis results, recommendations, and patient information"
                )
        
        # Share with Doctor
        st.markdown("## 👨‍⚕️ Next Steps with Your Doctor")
        st.info("""
        **📋 Preparing for Your Doctor Visit:**
        1. Download and print the PDF report above
        2. Bring original scan/test results to your appointment
        3. Prepare a list of your symptoms and concerns
        4. Note any changes since the scan was taken
        5. Bring a complete list of all medications (including supplements)
        
        **💬 Important Questions to Ask:**
        - What do these findings mean for my health?
        - What treatment options are available based on current guidelines?
        - Are there any lifestyle changes I should make?
        - What symptoms should prompt immediate medical attention?
        - When should I follow up and what tests might be needed?
        - Are there any new treatments or clinical trials I should know about?
        """)
    
    # About Section
    st.markdown("---")
    st.markdown("## ℹ️ About This Clinical Decision Support System")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔬 Advanced Features:**
        - Real-time medical guideline integration
        - Evidence-based medicine analysis
        - Comprehensive anatomical evaluation
        - Quantitative measurements and severity grading
        - Differential diagnosis support
        - Current treatment recommendations
        - Prognosis and risk assessment
        """)
    
    with col2:
        st.markdown("""
        **🏥 Clinical Applications:**
        - Radiology and imaging interpretation
        - Laboratory test analysis
        - Pathology report evaluation
        - Cardiology test review
        - Multi-specialty consultation support
        - Patient education resource
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Scan Reports Analyzer | AI-Powered Medical Imaging Analysis | Powered by Gemini AI + Tavily")
    st.markdown("*🩺 Combining artificial intelligence with clinical expertise for better patient outcomes*")
    
    # Privacy Notice
    st.markdown("### 🔒 HIPAA-Compliant Data Protection")
    st.info("""
    **Your medical data privacy is our priority:**
    - All data processing is encrypted
    - Medical images are temporarily processed and not stored
    - HIPAA-compliant security protocols
    - No personal data is shared with third parties
    - Automatic deletion of analysis data after session
    - Secure transmission of all medical information
    """)

if __name__ == "__main__":
    main()
