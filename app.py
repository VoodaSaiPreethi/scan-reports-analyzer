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
    
    .doctor-card {
        background-color: #F0FFF0;
        border: 1px solid #98FB98;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .condition-card {
        background-color: #F0F8FF;
        border: 1px solid #87CEFA;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
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

MEDICAL_SCAN_SYSTEM_PROMPT = """
You are an expert medical diagnostician and radiologist with extensive knowledge in medical imaging interpretation and patient care.
Your role is to analyze medical scan reports (X-rays, CT scans, MRIs, ultrasounds, blood tests, etc.) and provide comprehensive medical insights.

You must:
1. Identify and explain medical conditions found in the scan in simple, layman-friendly language
2. Assess severity and urgency of findings
3. Provide detailed explanations of what the conditions mean for the patient's health
4. Suggest appropriate precautions and lifestyle modifications
5. Determine if the condition requires immediate medical attention
6. Consider the patient's complete medical history, lifestyle, and family history in your analysis
7. Provide specific recommendations for specialist consultations when needed
8. List possible diseases or conditions that match the findings
"""

MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan report and provide a comprehensive analysis in the following structured format:

*Scan Analysis:* <detailed analysis of what the scan shows>
*Detected Conditions:* <list of medical conditions or abnormalities found>
*Condition Explanation:* <explain each condition in simple, understandable language>
*Severity Assessment:* <rate severity: Low/Moderate/High/Critical>
*Immediate Concerns:* <any urgent issues that need immediate attention>
*Recommended Precautions:* <specific precautions to take>
*Lifestyle Modifications:* <diet, exercise, habits to change>
*Specialist Consultations:* <which doctors to consult and when>
*Emergency Indicators:* <symptoms that require immediate medical help>
*Follow-up Requirements:* <when to get re-tested or follow-up scans>
*Prognosis:* <what to expect going forward>
*Possible Diseases:* <list of possible diseases that match these findings>

Always prioritize patient safety and provide clear, actionable advice.
"""

COMPREHENSIVE_ANALYSIS_PROMPT = """
You are a comprehensive medical advisor analyzing a patient's complete health profile including scan results, medical history, lifestyle, and family history.

Provide a detailed, personalized health assessment that considers:
- Current scan findings
- Patient's medical history and current medications
- Age-related health considerations
- Lifestyle factors (diet, exercise, habits)
- Family medical history and genetic predispositions
- Previous accidents or injuries
- Current symptoms and concerns

Deliver a holistic health report with personalized recommendations, risk assessments, and actionable health advice.
Include specific specialist recommendations and possible disease diagnoses based on all available information.
"""

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=MEDICAL_SCAN_SYSTEM_PROMPT,
            instructions=MEDICAL_ANALYSIS_INSTRUCTIONS,
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
            system_prompt=COMPREHENSIVE_ANALYSIS_PROMPT,
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
    """Analyze medical scan report using AI."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing medical scan report..."):
            response = agent.run(
                "Analyze this medical scan report and provide a comprehensive medical analysis including condition explanation, severity assessment, and recommendations.",
                images=[image_path],
            )
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error analyzing medical scan: {e}")
        return None

def comprehensive_health_analysis(scan_results, patient_info):
    """Perform comprehensive health analysis with patient information."""
    agent = get_comprehensive_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔍 Performing comprehensive health analysis..."):
            query = f"""
            Perform a comprehensive health analysis based on:
            
            SCAN RESULTS:
            {scan_results}
            
            PATIENT INFORMATION:
            {patient_info}
            
            Provide a detailed, personalized health assessment with specific recommendations, risk factors, and actionable advice.
            Consider all aspects of the patient's health profile in your analysis.
            Include specific specialist recommendations and possible disease diagnoses.
            """
            response = agent.run(query)
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error in comprehensive analysis: {e}")
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

def create_medical_pdf(image_data, scan_analysis, comprehensive_analysis, patient_info):
    """Create a comprehensive medical PDF report."""
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
            fontSize=20,
            alignment=1,
            spaceAfter=12,
            textColor=colors.HexColor('#8B4513')
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#A0522D'),
            spaceAfter=6
        )
        
        normal_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=11,
            leading=14
        )
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.red,
            borderWidth=1,
            borderColor=colors.red,
            borderPadding=5,
            backColor=colors.pink,
            alignment=1
        )
        
        # Title
        content.append(Paragraph("🏥 MediScan - Comprehensive Medical Report Analysis", title_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Medical disclaimer
        content.append(Paragraph(
            "⚠️ MEDICAL DISCLAIMER: This analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. "
            "Always consult with qualified healthcare professionals for medical decisions. In case of emergency, contact emergency services immediately.",
            disclaimer_style
        ))
        content.append(Spacer(1, 0.3*inch))
        
        # Report generation info
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"📅 Report Generated: {current_datetime}", normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Patient information
        content.append(Paragraph("👤 Patient Information:", heading_style))
        content.append(Paragraph(patient_info, normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Add scan image if available
        if image_data:
            try:
                img_temp = BytesIO(image_data)
                img = Image.open(img_temp)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                display_width = 4 * inch
                display_height = display_width * aspect
                
                img_temp.seek(0)
                img_obj = ReportLabImage(img_temp, width=display_width, height=display_height)
                content.append(Paragraph("📸 Medical Scan:", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.2*inch))
            except Exception as img_error:
                st.warning(f"Could not add image to PDF: {img_error}")
        
        # Scan analysis
        content.append(Paragraph("🔬 Initial Scan Analysis:", heading_style))
        content.append(Paragraph(scan_analysis.replace('<', '&lt;').replace('>', '&gt;'), normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Comprehensive analysis
        content.append(Paragraph("📊 Comprehensive Health Analysis:", heading_style))
        content.append(Paragraph(comprehensive_analysis.replace('<', '&lt;').replace('>', '&gt;'), normal_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Footer
        content.append(Paragraph("© 2025 MediScan Medical Report Analyzer | AI-Powered Medical Analysis", 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"📄 Error creating PDF: {e}")
        return None

def collect_patient_information():
    """Collect comprehensive patient information."""
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    with col2:
        occupation = st.text_input("Occupation", placeholder="e.g., Engineer, Teacher, etc.")
        smoker = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly", "Heavily"])
        exercise = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Medical History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("📋 Medical History")
    
    previous_conditions = st.text_area(
        "Previous Medical Conditions",
        placeholder="List any previous diagnoses, surgeries, or chronic conditions...",
        height=100
    )
    
    current_medications = st.text_area(
        "Current Medications",
        placeholder="List all current medications with dosages...",
        height=100
    )
    
    current_symptoms = st.text_area(
        "Current Symptoms",
        placeholder="Describe any symptoms you're currently experiencing...",
        height=100
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Family History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("👨‍👩‍👧‍👦 Family Medical History")
    
    family_history = st.text_area(
        "Family Medical History",
        placeholder="List any significant medical conditions in your family (parents, siblings, grandparents)...",
        height=100
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Accident History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🚑 Accident & Injury History")
    
    accident_history = st.text_area(
        "Previous Accidents or Injuries",
        placeholder="Describe any significant accidents, injuries, or trauma you've experienced...",
        height=100
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Diet and Lifestyle
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🥗 Diet & Lifestyle")
    
    diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Other"])
    diet_details = st.text_area(
        "Diet Details",
        placeholder="Describe your typical daily diet, eating habits, and any dietary restrictions...",
        height=80
    )
    
    sleep_hours = st.slider("Average Sleep Hours", 4, 12, 8)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compile patient information
    patient_info = f"""
    Age: {age} years
    Gender: {gender}
    Weight: {weight} kg
    Height: {height} cm
    Occupation: {occupation}
    Smoking Status: {smoker}
    Alcohol Consumption: {alcohol}
    Exercise Level: {exercise}
    
    Previous Medical Conditions: {previous_conditions}
    Current Medications: {current_medications}
    Current Symptoms: {current_symptoms}
    
    Family Medical History: {family_history}
    
    Accident/Injury History: {accident_history}
    
    Diet Type: {diet_type}
    Diet Details: {diet_details}
    Average Sleep: {sleep_hours} hours
    Stress Level: {stress_level}/10
    """
    
    return patient_info

def display_emergency_alert(content):
    """Display emergency alert with appropriate styling."""
    if any(keyword in content.lower() for keyword in ["emergency", "urgent", "immediate", "critical", "severe"]):
        st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
        st.error(f"🚨 EMERGENCY ALERT: {content}")
        st.markdown('</div>', unsafe_allow_html=True)
        return True
    return False

def display_doctor_recommendations(content):
    """Display doctor recommendations in a structured format."""
    doctors_db = {
        "Cardiologist": "Specializes in heart and cardiovascular system disorders",
        "Neurologist": "Treats brain and nervous system disorders",
        "Orthopedist": "Specializes in bones, joints, and muscles",
        "Rheumatologist": "Treats arthritis and autoimmune diseases",
        "Endocrinologist": "Specializes in hormonal and metabolic disorders",
        "Gastroenterologist": "Treats digestive system disorders",
        "Pulmonologist": "Specializes in lung and respiratory conditions",
        "Nephrologist": "Treats kidney diseases",
        "Hematologist": "Specializes in blood disorders",
        "Oncologist": "Diagnoses and treats cancer",
        "Dermatologist": "Treats skin conditions",
        "Urologist": "Specializes in urinary and male reproductive system",
        "Gynecologist": "Focuses on female reproductive health",
        "ENT Specialist": "Treats ear, nose, and throat conditions",
        "Ophthalmologist": "Specializes in eye diseases and vision problems",
        "Psychiatrist": "Treats mental health disorders",
        "Allergist/Immunologist": "Treats allergies and immune system disorders",
        "Infectious Disease Specialist": "Treats complex infections",
        "Geriatrician": "Specializes in elderly patient care",
        "Pediatrician": "Specializes in child healthcare"
    }
    
    recommended_doctors = []
    for doc in doctors_db:
        if doc.lower() in content.lower():
            recommended_doctors.append((doc, doctors_db[doc]))
    
    if recommended_doctors:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🏥 Recommended Specialist Doctors")
        st.markdown("Based on your condition, you should consider consulting these specialists:")
        
        for doc, desc in recommended_doctors:
            st.markdown(f"""
            <div class="doctor-card">
                <b>{doc}</b>: {desc}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ The analysis recommends specialist consultation but didn't specify which type. Please consult with your primary care physician for referral.")

def display_possible_conditions(content):
    """Display possible diseases/conditions in a structured format."""
    conditions_db = {
        "Hypertension": "High blood pressure that can lead to heart disease",
        "Diabetes": "Chronic condition affecting blood sugar regulation",
        "Arthritis": "Inflammation of joints causing pain and stiffness",
        "Osteoporosis": "Bone weakening increasing fracture risk",
        "COPD": "Chronic obstructive pulmonary disease (lung condition)",
        "Asthma": "Chronic inflammatory disease of the airways",
        "Coronary Artery Disease": "Narrowing of heart arteries",
        "Stroke": "Brain damage from interrupted blood supply",
        "Cancer": "Abnormal cell growth with potential to spread",
        "Kidney Disease": "Impaired kidney function",
        "Liver Disease": "Conditions affecting liver function",
        "Thyroid Disorder": "Overactive or underactive thyroid",
        "Anemia": "Low red blood cell count or hemoglobin",
        "Osteoarthritis": "Joint cartilage breakdown",
        "Rheumatoid Arthritis": "Autoimmune joint inflammation",
        "Gastritis": "Stomach lining inflammation",
        "Ulcerative Colitis": "Inflammatory bowel disease",
        "Diverticulitis": "Infection or inflammation of colon pouches",
        "Pneumonia": "Lung infection inflaming air sacs",
        "Bronchitis": "Inflammation of bronchial tubes",
        "Alzheimer's Disease": "Progressive brain disorder affecting memory",
        "Parkinson's Disease": "Neurological disorder affecting movement",
        "Multiple Sclerosis": "Autoimmune disease affecting central nervous system",
        "Lupus": "Autoimmune disease affecting multiple organs",
        "HIV/AIDS": "Viral infection compromising immune system",
        "Hepatitis": "Liver inflammation from viral infection",
        "Tuberculosis": "Bacterial infection primarily affecting lungs",
        "Malaria": "Parasitic disease transmitted by mosquitoes",
        "Dengue Fever": "Viral infection transmitted by mosquitoes",
        "COVID-19": "Respiratory illness caused by coronavirus"
    }
    
    detected_conditions = []
    for condition in conditions_db:
        if condition.lower() in content.lower():
            detected_conditions.append((condition, conditions_db[condition]))
    
    if detected_conditions:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Possible Diseases/Conditions")
        st.markdown("The following conditions may be present based on your scan:")
        
        for condition, desc in detected_conditions:
            st.markdown(f"""
            <div class="condition-card">
                <b>{condition}</b>: {desc}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ The analysis detected abnormalities but didn't specify exact conditions. Further diagnostic tests may be needed.")

def main():
    # Initialize session state
    if 'scan_analysis' not in st.session_state:
        st.session_state.scan_analysis = None
    if 'comprehensive_analysis' not in st.session_state:
        st.session_state.comprehensive_analysis = None
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None

    # Header
    st.markdown('<div class="linear-layout">', unsafe_allow_html=True)
    st.title("🏥 MediScan - Medical Report Analyzer")
    
    # Medical disclaimer
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.warning("""
    ⚠️ **MEDICAL DISCLAIMER**
    
    This AI-powered medical report analyzer is designed to provide educational information and preliminary insights only. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified 
    healthcare professionals for medical decisions. In case of emergency, contact emergency services immediately.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scan Upload Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📤 Upload Medical Scan Report")
    
    uploaded_file = st.file_uploader(
        "Upload your medical scan report (X-ray, CT, MRI, ultrasound, blood test, etc.)",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        help="Upload a clear image of your medical scan report or test results"
    )
    
    if uploaded_file:
        # Display uploaded image
        if uploaded_file.type.startswith('image/'):
            resized_image = resize_image_for_display(uploaded_file)
            if resized_image:
                st.image(resized_image, caption="Uploaded Medical Scan", width=MAX_IMAGE_WIDTH)
        
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB")
    
    # Patient Information Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    patient_info = collect_patient_information()
    
    # Analysis Button
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if uploaded_file and st.button("🔬 Analyze Medical Scan & Generate Report", key="analyze_button"):
        # Save uploaded file and analyze
        temp_path = save_uploaded_file(uploaded_file)
        if temp_path:
            try:
                # Initial scan analysis
                scan_analysis = analyze_medical_scan(temp_path)
                
                if scan_analysis:
                    st.session_state.scan_analysis = scan_analysis
                    st.session_state.patient_info = patient_info
                    st.session_state.original_image = uploaded_file.getvalue()
                    
                    # Comprehensive analysis
                    comprehensive_analysis = comprehensive_health_analysis(scan_analysis, patient_info)
                    if comprehensive_analysis:
                        st.session_state.comprehensive_analysis = comprehensive_analysis
                    
                    st.success("✅ Medical scan analysis completed successfully!")
                else:
                    st.error("❌ Analysis failed. Please try with a clearer image.")
                
            except Exception as e:
                st.error(f"🚨 Analysis failed: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Display Results
    if st.session_state.scan_analysis:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📊 Medical Analysis Results")
        
        # Parse and display initial scan analysis
        st.markdown("### 🔬 Initial Scan Analysis")
        analysis_text = st.session_state.scan_analysis
        
        # Check for emergency indicators
        emergency_found = False
        if "emergency" in analysis_text.lower() or "critical" in analysis_text.lower():
            emergency_found = True
        
        sections = [
            "Scan Analysis", "Detected Conditions", "Condition Explanation", 
            "Severity Assessment", "Immediate Concerns", "Recommended Precautions",
            "Lifestyle Modifications", "Specialist Consultations", "Emergency Indicators",
            "Follow-up Requirements", "Prognosis", "Possible Diseases"
        ]
        
        for section in sections:
            pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections)}):\*|$)"
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                
                # Icons for sections
                icons = {
                    "Scan Analysis": "🔍",
                    "Detected Conditions": "🩺",
                    "Condition Explanation": "📝",
                    "Severity Assessment": "⚖️",
                    "Immediate Concerns": "🚨",
                    "Recommended Precautions": "🛡️",
                    "Lifestyle Modifications": "🏃‍♂️",
                    "Specialist Consultations": "👨‍⚕️",
                    "Emergency Indicators": "🚨",
                    "Follow-up Requirements": "📅",
                    "Prognosis": "🔮",
                    "Possible Diseases": "📋"
                }
                
                st.markdown(f"**{icons.get(section, '📋')} {section}:**")
                
                # Special handling for emergency sections
                if section in ["Immediate Concerns", "Emergency Indicators"] and content:
                    display_emergency_alert(content)
                else:
                    st.write(content)
                
                # Enhanced display for Specialist Consultations
                if section == "Specialist Consultations" and content:
                    display_doctor_recommendations(content)
                
                # Enhanced display for Possible Diseases
                if section == "Possible Diseases" and content:
                    display_possible_conditions(content)
                
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
