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
    st.markdown("Please create a `.streamlit/secrets.toml` file with your `TAVILY_API_KEY` and `GOOGLE_API_KEY`.")
    st.stop()

MAX_IMAGE_WIDTH = 400

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
MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan report and provide a comprehensive analysis in the following structured format:

*Diagnosed Conditions:* <Bulleted list of ALL medical conditions identified with exact medical names>
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
"""

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical analysis agent."""
    try:
        # Using gemini-1.5-flash-latest for general availability and vision capabilities
        return Agent(
            model=Gemini(model="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY),
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
        # Using gemini-1.5-flash-latest for general availability
        return Agent(
            model=Gemini(model="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY),
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
        # Reset file pointer to the beginning before opening
        image_file.seek(0)
        img = Image.open(image_file)
        
        aspect_ratio = img.height / img.width
        new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
        img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        # Reset buffer pointer to the beginning for reading later
        buf.seek(0)
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
        with st.spinner("🔬 Analyzing medical scan report... This may take a moment."):
            # For Gemini's vision capability, ensure the image path points to an actual image file.
            # If the input is a PDF, the model will try to interpret it as an image of a document.
            response = agent.run(
                "Analyze this medical scan report and provide a comprehensive medical analysis including condition explanation, severity assessment, and recommendations. Focus on extracting information from the image.",
                images=[image_path], # Pass the path to the image file
            )
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error analyzing medical scan: {e}. Please ensure the uploaded file is a clear image of a report.")
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
            """
            response = agent.run(query)
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error in comprehensive analysis: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk."""
    try:
        # Use a more specific directory if needed, e.g., 'temp_uploads'
        # Ensure the directory exists
        # if not os.path.exists('temp_uploads'):
        #     os.makedirs('temp_uploads')
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if not file_extension: # Add a default if no extension is found
            file_extension = ".png" 
        
        # Using NamedTemporaryFile is good, but for phi-agents it's often better
        # to ensure the file is accessible by path.
        # This approach with NamedTemporaryFile is generally robust.
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
        # Replace newlines with <br/> for ReportLab Paragraphs
        formatted_patient_info = patient_info.replace("\n", "<br/>")
        content.append(Paragraph(formatted_patient_info, normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Add scan image if available
        if image_data:
            try:
                img_temp = BytesIO(image_data)
                img = Image.open(img_temp)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                display_width = 4 * inch # Max width for image in PDF
                display_height = display_width * aspect
                
                # Ensure image doesn't exceed page height or other constraints if needed
                if display_height > (letter[1] - 2 * 72 - 2 * inch): # Letter height - margins - some buffer
                    display_height = (letter[1] - 2 * 72 - 2 * inch)
                    display_width = display_height / aspect

                img_temp.seek(0)
                img_obj = ReportLabImage(img_temp, width=display_width, height=display_height)
                content.append(Paragraph("📸 Medical Scan:", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.2*inch))
            except Exception as img_error:
                st.warning(f"Could not add image to PDF: {img_error}")
        
        # Scan analysis
        content.append(Paragraph("🔬 Initial Scan Analysis:", heading_style))
        # It's better to process markdown to HTML or use a Markdown-to-ReportLab converter
        # For simplicity, here we replace common markdown elements.
        # This will need more robust parsing for complex markdown.
        processed_scan_analysis = scan_analysis.replace('**', '').replace('* ', '• ').replace('\n', '<br/>')
        content.append(Paragraph(processed_scan_analysis.replace('<', '&lt;').replace('>', '&gt;'), normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Comprehensive analysis
        content.append(Paragraph("📊 Comprehensive Health Analysis:", heading_style))
        processed_comprehensive_analysis = comprehensive_analysis.replace('**', '').replace('* ', '• ').replace('\n', '<br/>')
        content.append(Paragraph(processed_comprehensive_analysis.replace('<', '&lt;').replace('>', '&gt;'), normal_style))
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
        age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Your current age in years.")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Your biological gender.")
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, help="Your weight in kilograms.")
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, help="Your height in centimeters.")
    
    with col2:
        occupation = st.text_input("Occupation", placeholder="e.g., Engineer, Teacher, etc.", help="Your current profession.")
        smoker = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"], help="Indicate your smoking habits.")
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly", "Heavily"], help="How often do you consume alcohol?")
        exercise = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], help="How active are you physically?")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Medical History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("📋 Medical History")
    
    previous_conditions = st.text_area(
        "Previous Medical Conditions",
        placeholder="List any previous diagnoses, surgeries, or chronic conditions...",
        height=100,
        help="Provide details of any past medical conditions or operations."
    )
    
    current_medications = st.text_area(
        "Current Medications",
        placeholder="List all current medications with dosages...",
        height=100,
        help="Include prescription and over-the-counter medications, and supplements."
    )
    
    current_symptoms = st.text_area(
        "Current Symptoms",
        placeholder="Describe any symptoms you're currently experiencing...",
        height=100,
        help="Detail any current health concerns or symptoms."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Family History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("👨‍👩‍👧‍👦 Family Medical History")
    
    family_history = st.text_area(
        "Family Medical History",
        placeholder="List any significant medical conditions in your family (parents, siblings, grandparents)...",
        height=100,
        help="Mention conditions like heart disease, diabetes, cancer, etc., in your close relatives."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Accident History
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🚑 Accident & Injury History")
    
    accident_history = st.text_area(
        "Previous Accidents or Injuries",
        placeholder="Describe any significant accidents, injuries, or trauma you've experienced...",
        height=100,
        help="Include details about past accidents, fractures, or other injuries."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Diet and Lifestyle
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🥗 Diet & Lifestyle")
    
    diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Other"], help="What type of diet do you primarily follow?")
    diet_details = st.text_area(
        "Diet Details",
        placeholder="Describe your typical daily diet, eating habits, and any dietary restrictions...",
        height=80,
        help="Provide specifics about your meals, allergies, or restrictions."
    )
    
    sleep_hours = st.slider("Average Sleep Hours", 4, 12, 8, help="How many hours do you typically sleep per night?")
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, help="Rate your general stress level from 1 (low) to 10 (high).")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compile patient information
    patient_info = f"""
    Age: {age} years
    Gender: {gender}
    Weight: {weight} kg
    Height: {height} cm
    Occupation: {occupation if occupation else 'Not specified'}
    Smoking Status: {smoker}
    Alcohol Consumption: {alcohol}
    Exercise Level: {exercise}
    
    Previous Medical Conditions: {previous_conditions if previous_conditions else 'None'}
    Current Medications: {current_medications if current_medications else 'None'}
    Current Symptoms: {current_symptoms if current_symptoms else 'None'}
    
    Family Medical History: {family_history if family_history else 'None'}
    
    Accident/Injury History: {accident_history if accident_history else 'None'}
    
    Diet Type: {diet_type}
    Diet Details: {diet_details if diet_details else 'Not specified'}
    Average Sleep: {sleep_hours} hours
    Stress Level: {stress_level}/10
    """
    
    return patient_info

def display_emergency_alert(content):
    """Display emergency alert with appropriate styling."""
    # Using more specific keywords for better matching, and ensuring case-insensitivity
    if any(keyword in content.lower() for keyword in ["emergency", "urgent", "immediate", "critical", "life-threatening", "severe risk"]):
        st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
        st.error(f"🚨 EMERGENCY ALERT: {content}")
        st.markdown('</div>', unsafe_allow_html=True)
        return True
    return False

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
        "Upload a clear image of your medical scan report or test results (e.g., X-ray report, CT report, blood test results). Image formats: JPG, JPEG, PNG, WEBP. PDF uploads are treated as images.",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        help="Upload a clear image of your medical scan report or test results. For PDFs, ensure the content is readable as an image."
    )
    
    if uploaded_file:
        # Store original image bytes for PDF generation later
        st.session_state.original_image = uploaded_file.getvalue()

        # Display uploaded image
        # Use a BytesIO object for displaying, so the original_image can be read again
        if uploaded_file.type.startswith('image/') or uploaded_file.type == 'application/pdf':
            # Create a BytesIO from the original_image to display, as resize_image_for_display consumes it
            display_file = BytesIO(st.session_state.original_image)
            resized_image_bytes = resize_image_for_display(display_file)
            if resized_image_bytes:
                st.image(resized_image_bytes, caption="Uploaded Medical Scan", width=MAX_IMAGE_WIDTH)
        else:
            st.warning("Unsupported file type for direct display. Please upload an image or PDF.")
        
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB")
    
    # Patient Information Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    patient_info = collect_patient_information()
    # Store patient info in session state as it's collected
    st.session_state.patient_info = patient_info
    
    # Analysis Button
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🔬 Analyze Medical Scan & Generate Report", key="analyze_button"):
        if not uploaded_file:
            st.warning("Please upload a medical scan report to proceed.")
        else:
            # Save uploaded file and analyze
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                try:
                    # Initial scan analysis
                    scan_analysis = analyze_medical_scan(temp_path)
                    
                    if scan_analysis:
                        st.session_state.scan_analysis = scan_analysis
                        
                        # Comprehensive analysis
                        comprehensive_analysis = comprehensive_health_analysis(scan_analysis, st.session_state.patient_info)
                        if comprehensive_analysis:
                            st.session_state.comprehensive_analysis = comprehensive_analysis
                        else:
                            st.warning("Could not perform comprehensive analysis. Please check AI model status.")
                        
                        st.success("✅ Medical scan analysis completed successfully!")
                    else:
                        st.error("❌ Scan analysis failed. Please try with a clearer image or different file.")
                    
                except Exception as e:
                    st.error(f"🚨 An unexpected error occurred during analysis: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                st.error("Failed to save uploaded file for analysis.")

# Display Results
if st.session_state.scan_analysis:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 Medical Analysis Results")
    
    # Display Emergency Alert first if applicable
    if st.session_state.scan_analysis:
        display_emergency_alert(st.session_state.scan_analysis) # Check scan analysis for emergency keywords
    
    # Parse and display diagnosed conditions first
    st.markdown("### 🩺 Diagnosed Conditions")
    # Updated regex to be more robust for different endings
    conditions_match = re.search(r"\*Diagnosed Conditions:\*(.*?)(?=\*(Detailed Analysis|Condition Explanations|Severity Assessment|Immediate Concerns|Recommended Actions|Precautions|Lifestyle Modifications|Specialist Referrals|Emergency Indicators|Follow-up Plan|Prognosis):\*|$)", 
                               st.session_state.scan_analysis, re.DOTALL | re.IGNORECASE)
    
    if conditions_match:
        conditions_content = conditions_match.group(1).strip()
        if conditions_content: # Ensure content is not empty
            # Add special styling for conditions
            st.markdown(f"""
            <div style="background-color:#FFF0F5; padding:15px; border-radius:10px; border-left:5px solid #FF69B4">
            {conditions_content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No specific medical conditions were explicitly identified in the 'Diagnosed Conditions' section by the AI.")
    else:
        st.info("Could not extract 'Diagnosed Conditions' section. The AI output format might have varied.")
    
    st.markdown("---")
    
    # Rest of the analysis sections...
    sections = [
        "Detailed Analysis", "Condition Explanations", "Severity Assessment", 
        "Immediate Concerns", "Recommended Actions", "Precautions",
        "Lifestyle Modifications", "Specialist Referrals", "Emergency Indicators",
        "Follow-up Plan", "Prognosis"
    ]
    
    full_analysis_text = st.session_state.scan_analysis
    
    for i, section in enumerate(sections):
        # Adjusted regex to look for the next section or end of string
        # Using a lookahead for the next potential section or end of string
        next_section_pattern = ""
        if i < len(sections) - 1:
            next_section_pattern = "|".join(re.escape(s) for s in sections[i+1:])
            pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{next_section_pattern}):\*|$)"
        else: # For the last section, match till end of string
            pattern = rf"\*{re.escape(section)}:\*(.*)"
            
        match = re.search(pattern, full_analysis_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            if content: # Only display if content is not empty
                st.markdown(f"**{section.replace('_', ' ')}:**")
                
                if section == "Severity Assessment":
                    # Add color coding for severity
                    if "Critical" in content or "High" in content:
                        st.error(content)
                    elif "Moderate" in content:
                        st.warning(content)
                    else:
                        st.success(content)
                elif section == "Emergency Indicators":
                    st.markdown(f'<div class="emergency-alert">{content}</div>', unsafe_allow_html=True)
                else:
                    st.write(content)
                
                st.markdown("---")
            else:
                st.info(f"'{section.replace('_', ' ')}' section was found but contained no content.")
        # else: # Uncomment for debugging if sections are often missed
        #     st.warning(f"Could not find '{section.replace('_', ' ')}' section in the analysis.")

    # Display Comprehensive Health Analysis if available
    if st.session_state.comprehensive_analysis:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🌐 Comprehensive Health Analysis")
        st.write(st.session_state.comprehensive_analysis)
        st.markdown("---")

    # PDF download
    if st.session_state.original_image and st.session_state.scan_analysis:
        st.subheader("📄 Download Medical Report")
        
        pdf_bytes = create_medical_pdf(
            st.session_state.original_image,
            st.session_state.scan_analysis,
            st.session_state.comprehensive_analysis if st.session_state.comprehensive_analysis else "No comprehensive analysis was generated.",
            st.session_state.patient_info if st.session_state.patient_info else "Patient information not available."
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
        else:
            st.error("Failed to generate PDF report.")

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
