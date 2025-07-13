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
    page_title="MediScan - CT Scan Report Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🩺"
)

# Custom CSS for nude theme
st.markdown("""
<style>
    .main {
        background-color: #f7f3f0;
        color: #4a4a4a;
    }
    .stApp {
        background-color: #faf8f5;
    }
    .stTextInput > div > div > input {
        background-color: #f0ebe6;
        color: #4a4a4a;
        border: 1px solid #d4c4b0;
    }
    .stTextArea > div > div > textarea {
        background-color: #f0ebe6;
        color: #4a4a4a;
        border: 1px solid #d4c4b0;
    }
    .stSelectbox > div > div > select {
        background-color: #f0ebe6;
        color: #4a4a4a;
    }
    .stButton > button {
        background-color: #d4c4b0;
        color: #4a4a4a;
        border: 1px solid #c4b49d;
    }
    .stButton > button:hover {
        background-color: #c4b49d;
        color: #3a3a3a;
    }
    .stSidebar {
        background-color: #f0ebe6;
    }
    .stExpander {
        background-color: #f0ebe6;
        border: 1px solid #d4c4b0;
    }
    .stAlert {
        background-color: #ede7dc;
        border: 1px solid #d4c4b0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #6b5b4d;
    }
    .stMarkdown {
        color: #4a4a4a;
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

MAX_IMAGE_WIDTH = 600

SYSTEM_PROMPT = """
You are an expert medical professional specializing in radiology and CT scan interpretation with extensive knowledge in medical imaging analysis.
Your role is to analyze CT scan reports, interpret findings in simple layman terms, and provide comprehensive medical guidance.

You have access to current medical literature and guidelines to provide accurate, evidence-based interpretations.
Your analysis should be thorough, compassionate, and easy to understand for patients and their families.
"""

CT_ANALYSIS_INSTRUCTIONS = """
Analyze the CT scan report considering the patient's complete medical profile and provide a comprehensive assessment:

1. **SCAN ANALYSIS:**
   - Interpret CT scan findings in simple, understandable language
   - Explain what each finding means for the patient's health
   - Distinguish between normal variations and abnormal findings
   - Provide context based on patient's age, medical history, and symptoms

2. **ABNORMALITIES ASSESSMENT:**
   - Clearly identify and explain any abnormalities found
   - Rate severity levels (mild, moderate, severe)
   - Explain potential causes and implications
   - Discuss whether findings are acute or chronic

3. **MEDICAL CORRELATION:**
   - Connect findings with patient's medical history and current medications
   - Identify how lifestyle factors might contribute to findings
   - Assess impact of current health conditions on scan results

4. **RECOMMENDATIONS:**
   - Specify which medical specialists to consult
   - Provide urgency levels for consultations
   - Suggest immediate precautions and lifestyle modifications
   - Recommend follow-up imaging if needed

5. **LIFESTYLE GUIDANCE:**
   - Dietary recommendations based on findings
   - Exercise and activity modifications
   - Habit changes that could improve outcomes

Return analysis in structured format:
*Executive Summary:* <brief overview in simple terms>
*Detailed Findings:* <comprehensive explanation of all findings>
*Abnormalities Identified:* <specific abnormalities with severity>
*Medical Correlation:* <how findings relate to patient's health profile>
*Specialist Consultations:* <whom to see and urgency level>
*Immediate Precautions:* <urgent actions to take>
*Lifestyle Recommendations:* <diet, exercise, habits>
*Follow-up Requirements:* <future monitoring needs>
*Questions for Your Doctor:* <important questions to ask>
"""

@st.cache_resource
def get_ct_agent():
    """Initialize and cache the CT scan analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=SYSTEM_PROMPT,
            instructions=CT_ANALYSIS_INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"❌ Error initializing CT analysis agent: {e}")
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

def analyze_ct_scan(image_path, patient_data):
    """Analyze CT scan report with comprehensive patient data."""
    agent = get_ct_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing CT scan report and medical profile..."):
            query = f"""
            Analyze this CT scan report for a patient with the following profile:
            
            Medical History: {patient_data['medical_history']}
            Current Medications: {patient_data['medications']}
            Health Problems: {patient_data['health_problems']}
            Diet: {patient_data['diet']}
            Lifestyle: {patient_data['lifestyle']}
            Habits: {patient_data['habits']}
            Age: {patient_data['age']}
            Gender: {patient_data['gender']}
            
            Please provide a comprehensive analysis explaining findings in simple terms, identifying abnormalities, and providing medical guidance.
            """
            
            response = agent.run(query, images=[image_path])
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error analyzing CT scan: {e}")
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

def create_ct_pdf(image_data, analysis_results, patient_data):
    """Create a PDF report of the CT scan analysis."""
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
            spaceAfter=20,
            textColor=colors.HexColor('#6b5b4d')
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#6b5b4d'),
            spaceAfter=10
        )
        normal_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#4a4a4a')
        )
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            borderWidth=1,
            borderColor=colors.red,
            borderPadding=8,
            backColor=colors.HexColor('#ede7dc'),
            alignment=1
        )
        
        # Title
        content.append(Paragraph("🩺 MediScan - CT Scan Report Analysis", title_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Medical disclaimer
        content.append(Paragraph(
            "⚠️ MEDICAL DISCLAIMER: This analysis is for educational purposes only and should not replace professional medical advice. "
            "Always consult with qualified healthcare professionals for medical decisions and treatment plans.",
            disclaimer_style
        ))
        content.append(Spacer(1, 0.3*inch))
        
        # Patient information
        content.append(Paragraph("👤 Patient Information:", heading_style))
        content.append(Paragraph(f"Age: {patient_data['age']}", normal_style))
        content.append(Paragraph(f"Gender: {patient_data['gender']}", normal_style))
        content.append(Paragraph(f"Medical History: {patient_data['medical_history']}", normal_style))
        content.append(Paragraph(f"Current Medications: {patient_data['medications']}", normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"📅 Analysis Date: {current_datetime}", normal_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Add CT scan image
        if image_data:
            try:
                img_temp = BytesIO(image_data)
                img = Image.open(img_temp)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                display_width = 5 * inch
                display_height = display_width * aspect
                
                img_temp.seek(0)
                img_obj = ReportLabImage(img_temp, width=display_width, height=display_height)
                content.append(Paragraph("🖼️ CT Scan Image:", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.3*inch))
            except Exception as img_error:
                st.warning(f"Could not add image to PDF: {img_error}")
        
        # Analysis results
        content.append(Paragraph("📊 Detailed Analysis Results:", heading_style))
        
        if analysis_results:
            sections = [
                "Executive Summary", "Detailed Findings", "Abnormalities Identified",
                "Medical Correlation", "Specialist Consultations", "Immediate Precautions",
                "Lifestyle Recommendations", "Follow-up Requirements", "Questions for Your Doctor"
            ]
            
            for section in sections:
                pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections)}):\*|$)"
                match = re.search(pattern, analysis_results, re.DOTALL | re.IGNORECASE)
                
                if match:
                    section_content = match.group(1).strip()
                    content.append(Paragraph(f"{section}:", heading_style))
                    
                    paragraphs = section_content.split("\n")
                    for para in paragraphs:
                        if para.strip():
                            clean_para = para.strip().replace('<', '&lt;').replace('>', '&gt;')
                            content.append(Paragraph(clean_para, normal_style))
                    
                    content.append(Spacer(1, 0.2*inch))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph("© 2025 MediScan - CT Scan Report Analyzer | Powered by Gemini AI + Tavily", 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"📄 Error creating PDF: {e}")
        return None

def display_analysis_section(title, content, icon):
    """Display analysis section with proper formatting."""
    if content:
        st.markdown(f"### {icon} {title}")
        
        # Special formatting for different sections
        if title == "Abnormalities Identified":
            if "severe" in content.lower():
                st.error(f"🚨 {content}")
            elif "moderate" in content.lower():
                st.warning(f"⚠️ {content}")
            elif "mild" in content.lower():
                st.info(f"ℹ️ {content}")
            else:
                st.write(content)
        elif title == "Specialist Consultations":
            if "urgent" in content.lower() or "immediate" in content.lower():
                st.error(f"🚨 {content}")
            elif "soon" in content.lower():
                st.warning(f"⚠️ {content}")
            else:
                st.info(f"👨‍⚕️ {content}")
        elif title == "Immediate Precautions":
            st.warning(f"⚠️ {content}")
        else:
            st.write(content)
        
        st.markdown("---")

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'ct_image' not in st.session_state:
        st.session_state.ct_image = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None

    # Header
    st.title("🩺 MediScan - CT Scan Report Analyzer")
    st.markdown("### Advanced AI-Powered Medical Imaging Analysis")
    
    # Medical disclaimer
    st.error("""
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
    
    This CT scan analysis tool is designed for educational and informational purposes only. The interpretation provided should NOT replace professional medical advice, diagnosis, or treatment. Always consult with qualified radiologists and healthcare professionals for accurate medical assessment and treatment decisions.
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
        placeholder="e.g., Diabetes (2015), Hypertension (2018), Appendectomy (2020), Family history of heart disease...",
        height=100
    )
    
    # Current Medications
    st.markdown("### 💊 Current Medications")
    medications = st.text_area(
        "List all current medications with dosages:",
        placeholder="e.g., Metformin 500mg twice daily, Lisinopril 10mg once daily, Aspirin 81mg daily...",
        height=100
    )
    
    # Health Problems
    st.markdown("### 🏥 Current Health Problems")
    health_problems = st.text_area(
        "Current symptoms and health concerns:",
        placeholder="e.g., Chest pain, shortness of breath, chronic fatigue, abdominal pain...",
        height=100
    )
    
    # Diet and Lifestyle
    st.markdown("### 🥗 Diet and Lifestyle")
    diet = st.text_area(
        "Dietary habits and preferences:",
        placeholder="e.g., Vegetarian, high sodium intake, irregular meals, fast food consumption...",
        height=80
    )
    
    lifestyle = st.text_area(
        "Lifestyle factors:",
        placeholder="e.g., Sedentary work, regular exercise 3x/week, high stress levels, irregular sleep...",
        height=80
    )
    
    # Habits
    st.markdown("### 🚬 Habits")
    habits = st.text_area(
        "Smoking, alcohol consumption, and other relevant habits:",
        placeholder="e.g., Former smoker (quit 2 years ago), occasional alcohol (2-3 drinks/week), no recreational drugs...",
        height=80
    )
    
    # CT Scan Upload
    st.markdown("## 🖼️ CT Scan Report Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CT scan report image (DICOM, JPG, PNG, or PDF scan):",
        type=["jpg", "jpeg", "png", "webp", "pdf", "dcm"],
        help="Upload a clear image of your CT scan report or the actual scan images"
    )
    
    if uploaded_file:
        # Display uploaded image
        if uploaded_file.type.startswith('image/'):
            resized_image = resize_image_for_display(uploaded_file)
            if resized_image:
                st.image(resized_image, caption="Uploaded CT Scan Report", width=MAX_IMAGE_WIDTH)
        
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB")
    
    # Analysis Button
    if uploaded_file and st.button("🔬 Analyze CT Scan Report", type="primary"):
        # Prepare patient data
        patient_data = {
            'age': age,
            'gender': gender,
            'medical_history': medical_history,
            'medications': medications,
            'health_problems': health_problems,
            'diet': diet,
            'lifestyle': lifestyle,
            'habits': habits
        }
        
        # Save uploaded file and analyze
        temp_path = save_uploaded_file(uploaded_file)
        if temp_path:
            try:
                analysis_result = analyze_ct_scan(temp_path, patient_data)
                
                if analysis_result:
                    st.session_state.analysis_results = analysis_result
                    st.session_state.ct_image = uploaded_file.getvalue()
                    st.session_state.patient_data = patient_data
                    
                    st.success("✅ CT scan analysis completed successfully!")
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
        st.markdown("## 📊 CT Scan Analysis Results")
        
        analysis_text = st.session_state.analysis_results
        
        # Define sections with appropriate icons
        sections = {
            "Executive Summary": "📋",
            "Detailed Findings": "🔍",
            "Abnormalities Identified": "⚠️",
            "Medical Correlation": "🔗",
            "Specialist Consultations": "👨‍⚕️",
            "Immediate Precautions": "🚨",
            "Lifestyle Recommendations": "🌟",
            "Follow-up Requirements": "📅",
            "Questions for Your Doctor": "❓"
        }
        
        for section, icon in sections.items():
            pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections.keys())}):\*|$)"
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                display_analysis_section(section, content, icon)
        
        # Emergency Contact Information
        st.markdown("## 🚨 Emergency Contact Information")
        st.error("""
        **If you experience any of the following, seek immediate medical attention:**
        - Severe chest pain or pressure
        - Difficulty breathing or shortness of breath
        - Sudden severe headache
        - Loss of consciousness
        - Severe abdominal pain
        - Signs of stroke (sudden weakness, speech difficulties, facial drooping)
        
        **Emergency Services:** Call your local emergency number (911, 108, etc.)
        """)
        
        # PDF Download
        st.markdown("## 📄 Download Complete Report")
        
        if st.session_state.ct_image and st.session_state.patient_data:
            pdf_bytes = create_ct_pdf(
                st.session_state.ct_image,
                st.session_state.analysis_results,
                st.session_state.patient_data
            )
            if pdf_bytes:
                download_filename = f"ct_scan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥 Download Complete CT Analysis Report",
                    data=pdf_bytes,
                    file_name=download_filename,
                    mime="application/pdf",
                    help="Download a comprehensive PDF report with all analysis results and recommendations"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 MediScan - CT Scan Report Analyzer | Powered by Gemini AI + Tavily")
    st.markdown("*For educational purposes only. Not a substitute for professional medical advice.*")

if __name__ == "__main__":
    main()
