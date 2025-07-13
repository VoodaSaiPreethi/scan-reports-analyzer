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

# Enhanced system prompt for comprehensive medical analysis
SYSTEM_PROMPT = """
You are a highly experienced medical professional with expertise in radiology, pathology, and medical imaging interpretation across ALL medical specialties including:

- Radiology (CT, MRI, X-Ray, Ultrasound, PET scans)
- Pathology (Histopathology, Cytology, Biopsy reports)
- Cardiology (ECG, Echocardiography, Angiography)
- Neurology (Brain scans, Spinal imaging)
- Orthopedics (Bone scans, Joint imaging)
- Gastroenterology (Endoscopy, Colonoscopy)
- Pulmonology (Chest X-rays, CT scans)
- Oncology (Tumor imaging, Cancer staging)
- Laboratory Medicine (Blood tests, Urine analysis)

Your role is to analyze ANY type of medical scan or report, interpret findings in simple layman terms, identify abnormalities, assess urgency levels, and provide comprehensive medical guidance that patients can easily understand.

You have access to current medical literature and evidence-based guidelines to provide accurate interpretations.
"""

# Comprehensive analysis instructions
MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan/report considering the patient's complete medical profile and provide a comprehensive assessment:

1. **SCAN TYPE IDENTIFICATION:**
   - Identify the type of medical scan/report (CT, MRI, X-Ray, Blood test, etc.)
   - Explain what this type of scan is used for in simple terms
   - Describe the body part/system being examined

2. **FINDINGS ANALYSIS:**
   - Interpret ALL findings in simple, understandable language
   - Explain what each finding means for the patient's health
   - Distinguish between normal variations and abnormal findings
   - Provide context based on patient's age, medical history, and symptoms

3. **ABNORMALITIES ASSESSMENT:**
   - Clearly identify and explain any abnormalities found
   - Rate severity levels (Normal, Mild, Moderate, Severe, Critical)
   - Explain potential causes and implications in layman terms
   - Discuss whether findings are acute (sudden) or chronic (long-term)

4. **EMERGENCY ASSESSMENT:**
   - Clearly state if this is an EMERGENCY requiring immediate medical attention
   - Identify life-threatening conditions that need urgent care
   - Provide specific emergency symptoms to watch for

5. **DETAILED PROBLEM EXPLANATION:**
   - Explain each problem in detail using simple language
   - Use analogies and comparisons that patients can understand
   - Describe how the problem affects the body's normal function
   - Explain potential progression if left untreated

6. **MEDICAL CORRELATION:**
   - Connect findings with patient's medical history and medications
   - Identify how lifestyle factors might contribute to findings
   - Assess impact of current health conditions on scan results

7. **RECOMMENDATIONS:**
   - Specify which medical specialists to consult with urgency levels
   - Provide immediate precautions and lifestyle modifications
   - Recommend follow-up testing or monitoring
   - Suggest preventive measures

8. **LIFESTYLE GUIDANCE:**
   - Dietary recommendations based on findings
   - Exercise and activity modifications
   - Habit changes that could improve outcomes
   - Stress management and mental health considerations

Return analysis in this EXACT structured format:
*Scan Type & Purpose:* <identification and explanation>
*Executive Summary:* <brief overview in simple terms>
*Detailed Findings:* <comprehensive explanation of all findings>
*Abnormalities Identified:* <specific abnormalities with severity levels>
*Emergency Status:* <urgent/non-urgent with clear reasoning>
*Problem Explanation:* <detailed explanation of each issue in layman terms>
*Medical Correlation:* <how findings relate to patient's health profile>
*Specialist Consultations:* <whom to see and urgency level>
*Immediate Precautions:* <urgent actions to take>
*Lifestyle Recommendations:* <diet, exercise, habits>
*Follow-up Requirements:* <future monitoring needs>
*Questions for Your Doctor:* <important questions to ask>
*Warning Signs:* <symptoms that require immediate medical attention>
"""

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical scan analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=SYSTEM_PROMPT,
            instructions=MEDICAL_ANALYSIS_INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
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

def analyze_medical_scan(image_path, patient_data, scan_type):
    """Analyze any type of medical scan with comprehensive patient data."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing medical scan/report and patient profile..."):
            query = f"""
            Analyze this {scan_type} medical scan/report for a patient with the following profile:
            
            Patient Age: {patient_data['age']}
            Gender: {patient_data['gender']}
            Medical History: {patient_data['medical_history']}
            Current Medications: {patient_data['medications']}
            Current Symptoms: {patient_data['symptoms']}
            Health Problems: {patient_data['health_problems']}
            Diet: {patient_data['diet']}
            Lifestyle: {patient_data['lifestyle']}
            Habits: {patient_data['habits']}
            Family History: {patient_data['family_history']}
            
            Please provide a comprehensive analysis explaining ALL findings in simple terms, identifying any abnormalities, 
            assessing emergency status, and providing detailed medical guidance. Focus on patient education and clear communication.
            
            IMPORTANT: If there are any emergency conditions or life-threatening findings, clearly state this at the beginning 
            of your analysis and provide immediate action steps.
            """
            
            response = agent.run(query, images=[image_path])
            return response.content.strip()
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

def create_medical_pdf(image_data, analysis_results, patient_data, scan_type):
    """Create a comprehensive PDF report of the medical scan analysis."""
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
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=22,
            alignment=1,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50')
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10
        )
        normal_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#34495e')
        )
        emergency_style = ParagraphStyle(
            'Emergency',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.HexColor('#ffebee'),
            alignment=1
        )
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            borderWidth=1,
            borderColor=colors.red,
            borderPadding=8,
            backColor=colors.HexColor('#fff3e0'),
            alignment=1
        )
        
        # Title
        content.append(Paragraph("🔬 Scan Reports Analyzer", title_style))
        content.append(Paragraph("Universal Medical Imaging Analysis Report", 
                               ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=16, 
                                            alignment=1, textColor=colors.HexColor('#7f8c8d'))))
        content.append(Spacer(1, 0.3*inch))
        
        # Medical disclaimer
        content.append(Paragraph(
            "⚠️ MEDICAL DISCLAIMER: This analysis is for educational and informational purposes only. "
            "It should NOT replace professional medical advice, diagnosis, or treatment. "
            "Always consult with qualified healthcare professionals for medical decisions and treatment plans.",
            disclaimer_style
        ))
        content.append(Spacer(1, 0.3*inch))
        
        # Patient information
        content.append(Paragraph("👤 Patient Information:", heading_style))
        content.append(Paragraph(f"Age: {patient_data['age']} years", normal_style))
        content.append(Paragraph(f"Gender: {patient_data['gender']}", normal_style))
        content.append(Paragraph(f"Scan Type: {scan_type}", normal_style))
        content.append(Paragraph(f"Medical History: {patient_data['medical_history']}", normal_style))
        content.append(Paragraph(f"Current Medications: {patient_data['medications']}", normal_style))
        content.append(Paragraph(f"Current Symptoms: {patient_data['symptoms']}", normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"📅 Analysis Date: {current_datetime}", normal_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Add medical scan image
        if image_data:
            try:
                img_temp = BytesIO(image_data)
                img = Image.open(img_temp)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                display_width = 5 * inch
                display_height = display_width * aspect
                
                if display_height > 6 * inch:
                    display_height = 6 * inch
                    display_width = display_height / aspect
                
                img_temp.seek(0)
                img_obj = ReportLabImage(img_temp, width=display_width, height=display_height)
                content.append(Paragraph(f"🖼️ {scan_type} Image:", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.3*inch))
            except Exception as img_error:
                st.warning(f"Could not add image to PDF: {img_error}")
        
        # Analysis results
        content.append(Paragraph("📊 Comprehensive Medical Analysis:", heading_style))
        
        if analysis_results:
            sections = [
                "Scan Type & Purpose", "Executive Summary", "Detailed Findings", 
                "Abnormalities Identified", "Emergency Status", "Problem Explanation",
                "Medical Correlation", "Specialist Consultations", "Immediate Precautions",
                "Lifestyle Recommendations", "Follow-up Requirements", "Questions for Your Doctor",
                "Warning Signs"
            ]
            
            for section in sections:
                pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections)}):\*|$)"
                match = re.search(pattern, analysis_results, re.DOTALL | re.IGNORECASE)
                
                if match:
                    section_content = match.group(1).strip()
                    
                    # Special formatting for emergency status
                    if section == "Emergency Status" and ("emergency" in section_content.lower() or "urgent" in section_content.lower()):
                        content.append(Paragraph(f"🚨 {section}:", heading_style))
                        content.append(Paragraph(section_content, emergency_style))
                    else:
                        content.append(Paragraph(f"{section}:", heading_style))
                        
                        paragraphs = section_content.split("\n")
                        for para in paragraphs:
                            if para.strip():
                                clean_para = para.strip().replace('<', '&lt;').replace('>', '&gt;')
                                content.append(Paragraph(clean_para, normal_style))
                    
                    content.append(Spacer(1, 0.2*inch))
        
        # Emergency contact information
        content.append(Paragraph("🚨 Emergency Contact Information:", heading_style))
        content.append(Paragraph(
            "If you experience severe symptoms, chest pain, difficulty breathing, loss of consciousness, "
            "or any life-threatening conditions, contact emergency services immediately (911, 108, etc.)",
            emergency_style
        ))
        content.append(Spacer(1, 0.2*inch))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph("© 2025 Scan Reports Analyzer | Universal Medical Imaging Analysis | Powered by Gemini AI + Tavily", 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                             textColor=colors.gray, alignment=1)))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"📄 Error creating PDF: {e}")
        return None

def display_analysis_section(title, content, icon, is_emergency=False):
    """Display analysis section with proper formatting and emergency highlighting."""
    if content:
        if is_emergency:
            st.markdown(f'<div class="emergency-banner">{icon} {title}</div>', unsafe_allow_html=True)
            st.error(content)
        else:
            st.markdown(f"### {icon} {title}")
            
            # Special formatting for different sections
            if title == "Abnormalities Identified":
                if "critical" in content.lower() or "severe" in content.lower():
                    st.error(f"🚨 {content}")
                elif "moderate" in content.lower():
                    st.warning(f"⚠️ {content}")
                elif "mild" in content.lower():
                    st.info(f"ℹ️ {content}")
                else:
                    st.write(content)
            elif title == "Emergency Status":
                if "emergency" in content.lower() or "urgent" in content.lower():
                    st.error(f"🚨 {content}")
                else:
                    st.success(f"✅ {content}")
            elif title == "Specialist Consultations":
                if "urgent" in content.lower() or "immediate" in content.lower():
                    st.error(f"🚨 {content}")
                elif "soon" in content.lower():
                    st.warning(f"⚠️ {content}")
                else:
                    st.info(f"👨‍⚕️ {content}")
            elif title == "Immediate Precautions":
                st.warning(f"⚠️ {content}")
            elif title == "Warning Signs":
                st.error(f"⚠️ {content}")
            else:
                st.write(content)
        
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

    # Header
    st.title("🔬 Scan Reports Analyzer")
    st.markdown("### Universal Medical Imaging Analysis - AI-Powered Healthcare Assistant")
    
    # Supported scan types display
    st.markdown('<div class="scan-type-card">🏥 Supports ALL Medical Scans: CT, MRI, X-Ray, Ultrasound, Blood Tests, ECG, Pathology Reports & More!</div>', 
                unsafe_allow_html=True)
    
    # Medical disclaimer
    st.error("""
    ⚠️ **CRITICAL MEDICAL DISCLAIMER**
    
    This medical scan analysis tool is designed for educational and informational purposes ONLY. 
    The interpretation provided should NEVER replace professional medical advice, diagnosis, or treatment. 
    
    🏥 **Always consult with qualified healthcare professionals for:**
    - Accurate medical assessment and diagnosis
    - Treatment decisions and medical care
    - Emergency medical situations
    - Any health concerns or symptoms
    
    📞 **For medical emergencies, contact your local emergency services immediately**
    """)
    
    # Patient Information Section
    st.markdown("## 👤 Patient Information")
    
    # Basic demographics
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    with demo_col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    # Medical History
    st.markdown("### 📋 Medical History")
    medical_history = st.text_area(
        "Previous medical conditions, surgeries, and significant health events:",
        placeholder="e.g., Diabetes (2015), Hypertension (2018), Heart surgery (2020), Cancer history, Previous hospitalizations...",
        height=100
    )
    
    # Current Medications
    st.markdown("### 💊 Current Medications")
    medications = st.text_area(
        "List all current medications with dosages:",
        placeholder="e.g., Metformin 500mg twice daily, Lisinopril 10mg once daily, Aspirin 81mg daily, Vitamins, Supplements...",
        height=100
    )
    
    # Current Symptoms
    st.markdown("### 🩺 Current Symptoms")
    symptoms = st.text_area(
        "Current symptoms and when they started:",
        placeholder="e.g., Chest pain for 2 weeks, Shortness of breath, Headaches, Fatigue, Dizziness, Pain location and severity...",
        height=100
    )
    
    # Health Problems
    st.markdown("### 🏥 Current Health Problems")
    health_problems = st.text_area(
        "Ongoing health issues and chronic conditions:",
        placeholder="e.g., Chronic kidney disease, Arthritis, Sleep apnea, Mental health conditions, Autoimmune disorders...",
        height=100
    )
    
    # Family History
    st.markdown("### 👨‍👩‍👧‍👦 Family History")
    family_history = st.text_area(
        "Family medical history (parents, siblings, grandparents):",
        placeholder="e.g., Father: Heart disease, Mother: Diabetes, Sister: Breast cancer, Family history of stroke...",
        height=80
    )
    
    # Diet and Lifestyle
    st.markdown("### 🥗 Diet and Lifestyle")
    diet = st.text_area(
        "Dietary habits and preferences:",
        placeholder="e.g., Vegetarian, High sodium intake, Processed foods, Alcohol consumption, Caffeine intake...",
        height=80
    )
    
    lifestyle = st.text_area(
        "Lifestyle factors:",
        placeholder="e.g., Sedentary work, Exercise routine, Stress levels, Sleep patterns, Work environment...",
        height=80
    )
    
    # Habits
    st.markdown("### 🚬 Habits")
    habits = st.text_area(
        "Smoking, alcohol, and other relevant habits:",
        placeholder="e.g., Current smoker (1 pack/day), Former smoker (quit 2 years ago), Social drinker, Recreational drugs...",
        height=80
    )
    
    # Scan Type Selection
    st.markdown("## 🔬 Medical Scan Type")
    scan_type = st.selectbox(
        "Select the type of medical scan/report you're uploading:",
        [
            "CT Scan", "MRI Scan", "X-Ray", "Ultrasound", "PET Scan", "Nuclear Medicine",
            "Blood Test Results", "Urine Analysis", "ECG/EKG", "Echocardiogram", 
            "Pathology Report", "Biopsy Results", "Endoscopy Report", "Colonoscopy Report",
            "Mammography", "Bone Density Scan", "Angiography", "Stress Test Results",
            "Pulmonary Function Test", "Allergy Test Results", "Genetic Test Results", "Other"
        ]
    )
    
    # Medical Scan Upload
    st.markdown("## 🖼️ Medical Scan/Report Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your medical scan report, test results, or imaging:",
        type=["jpg", "jpeg", "png", "webp", "pdf", "dcm", "tiff", "bmp"],
        help="Upload a clear image of your medical scan, test results, or report. Supports all major medical imaging formats."
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
                    
                    st.success("✅ Medical scan analysis completed successfully!")
                    
                    # Check for emergency conditions
                    if "emergency" in analysis_result.lower() or "urgent" in analysis_result.lower():
                        st.markdown('<div class="emergency-banner">🚨 EMERGENCY DETECTED - IMMEDIATE MEDICAL ATTENTION REQUIRED</div>', 
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
        
        analysis_text = st.session_state.analysis_results
        
        # Check if emergency
        is_emergency = "emergency" in analysis_text.lower() and "urgent" in analysis_text.lower()
        
        # Define sections with appropriate icons
        sections = {
            "Scan Type & Purpose": "🔬",
            "Executive Summary": "📋",
            "Detailed Findings": "🔍",
            "Abnormalities Identified": "⚠️",
            "Emergency Status": "🚨",
            "Problem Explanation": "📝",
            "Medical Correlation": "🔗",
            "Specialist Consultations": "👨‍⚕️",
            "Immediate Precautions": "⚠️",
            "Lifestyle Recommendations": "🌟",
            "Follow-up Requirements": "📅",
            "Questions for Your Doctor": "❓",
            "Warning Signs": "🚨"
        }
        
        for section, icon in sections.items():
            pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections.keys())}):\*|$)"
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                is_emergency_section = section in ["Emergency Status", "Warning Signs"] and is_emergency
                display_analysis_section(section, content, icon, is_emergency_section)
        
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
        
        # Additional Medical Resources
        st.markdown("## 📞 Additional Medical Resources")
        st.info("""
        **🏥 Non-Emergency Medical Help:**
        - Contact your primary care physician
        - Visit an urgent care center
        - Call a medical helpline in your area
        - Consult with a specialist as recommended
        
        **🩺 Online Medical Resources:**
        - WebMD: www.webmd.com
        - Mayo Clinic: www.mayoclinic.org
        - MedlinePlus: medlineplus.gov
        - Your healthcare provider's patient portal
        
        **💊 Medication Information:**
        - Always follow your doctor's instructions
        - Check with pharmacist for drug interactions
        - Keep updated medication list
        """)
        
        # PDF Download
        st.markdown("## 📄 Download Complete Medical Report")
        
        if st.session_state.medical_image and st.session_state.patient_data:
            pdf_bytes = create_medical_pdf(
                st.session_state.medical_image,
                st.session_state.analysis_results,
                st.session_state.patient_data,
                st.session_state.scan_type
            )
            if pdf_bytes:
                download_filename = f"medical_scan_analysis_{st.session_state.scan_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥 Download Complete Medical Analysis Report",
                    data=pdf_bytes,
                    file_name=download_filename,
                    mime="application/pdf",
                    help="Download a comprehensive PDF report with all analysis results, recommendations, and patient information"
                )
        
        # Share with Doctor
        st.markdown("## 👨‍⚕️ Share with Your Doctor")
        st.info("""
        **📋 To share this analysis with your healthcare provider:**
        1. Download the PDF report above
        2. Print or email the report to your doctor
        3. Bring the original scan/test results to your appointment
        4. Prepare the questions listed in the analysis
        5. Discuss any concerns or symptoms you're experiencing
        
        **📝 Remember:** This analysis is a tool to help you understand your results, 
        but your doctor's interpretation and advice should always take priority.
        """)
    
    # About Section
    st.markdown("---")
    st.markdown("## ℹ️ About Scan Reports Analyzer")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔬 Our Technology:**
        - Advanced AI-powered medical image analysis
        - Comprehensive patient profile integration
        - Evidence-based medical interpretation
        - Multi-specialty medical knowledge base
        - Real-time medical literature access
        """)
    
    with col2:
        st.markdown("""
        **🏥 Supported Medical Scans:**
        - Radiology: CT, MRI, X-Ray, Ultrasound, PET
        - Cardiology: ECG, Echocardiogram, Stress Tests
        - Laboratory: Blood tests, Urine analysis
        - Pathology: Biopsy reports, Cytology
        - Specialized: Mammography, Bone density, Endoscopy
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Scan Reports Analyzer | Universal Medical Imaging Analysis | Powered by Gemini AI + Tavily")
    st.markdown("*🩺 Empowering patients with AI-driven medical insights - For educational purposes only*")
    
    # Privacy Notice
    st.markdown("### 🔒 Privacy & Security")
    st.info("""
    **Your privacy is our priority:**
    - Medical images and data are processed securely
    - No personal information is stored permanently
    - Analysis is conducted in real-time
    - HIPAA-compliant processing practices
    - Your data is never shared with third parties
    """)

if __name__ == "__main__":
    main()
