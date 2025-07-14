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
    page_title="Scan Reports Analyzer - Medical Scan Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔬"
)

# API Keys
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Check if API keys are available
if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    st.error("🔑 API keys are missing. Please check your configuration.")
    st.stop()

MAX_IMAGE_WIDTH = 400

MEDICAL_SCAN_SYSTEM_PROMPT = """
You are an expert medical imaging specialist and radiologist with advanced knowledge in analyzing all types of medical scans and identifying abnormalities.
Your role is to analyze medical scans (X-rays, CT scans, MRIs, ultrasounds, mammograms, bone scans, etc.) and provide comprehensive analysis in simple, understandable language.

Your expertise includes:
- Identifying abnormalities, lesions, fractures, tumors, and other medical conditions
- Explaining findings in layman's terms that patients can understand
- Determining urgency levels and when immediate medical attention is required
- Providing detailed explanations of what the abnormalities mean for the patient's health
- Suggesting appropriate next steps and specialist consultations

Always prioritize patient safety and provide clear, accurate, and compassionate analysis.
"""

SCAN_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan image thoroughly and provide a comprehensive report with the following structure:

1. **Scan Type Identification**: Identify what type of medical scan this is (X-ray, CT, MRI, ultrasound, etc.)

2. **Primary Findings**: List all abnormalities, irregularities, or concerning findings you observe

3. **Detailed Explanation**: Explain each finding in simple, non-technical language that a patient can understand

4. **Urgency Assessment**: Determine if this is:
   - EMERGENCY: Requires immediate hospital attention
   - URGENT: Needs medical consultation within 24-48 hours
   - ROUTINE: Can be discussed at next scheduled appointment
   - NORMAL: No significant abnormalities detected

5. **What This Means for You**: Explain in layman's terms what these findings could mean for the patient's health

6. **Recommended Actions**: Provide specific recommendations for next steps

7. **Areas of Concern**: Highlight any specific areas that need attention

8. **Follow-up Requirements**: Suggest what type of follow-up or additional tests might be needed

Format your response clearly with appropriate headings and use simple, compassionate language throughout.
If you identify any emergency conditions, clearly state "🚨 EMERGENCY - SEEK IMMEDIATE MEDICAL ATTENTION" at the beginning.
"""

EMERGENCY_CONDITIONS = [
    "pneumothorax", "pulmonary embolism", "aortic dissection", "stroke", "brain hemorrhage",
    "acute appendicitis", "bowel obstruction", "kidney stones", "fractures", "internal bleeding",
    "tumor", "mass", "aneurysm", "heart attack", "acute", "severe", "critical"
]

@st.cache_resource
def get_scan_analysis_agent():
    """Initialize and cache the medical scan analysis agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=MEDICAL_SCAN_SYSTEM_PROMPT,
            instructions=SCAN_ANALYSIS_INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"❌ Error initializing scan analysis agent: {e}")
        return None

@st.cache_resource
def get_second_opinion_agent():
    """Initialize and cache the second opinion agent for verification."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt="""You are a senior radiologist providing second opinion on medical scan analysis. 
            Review the initial analysis and provide additional insights, corrections, or confirmations. 
            Focus on accuracy and patient safety.""",
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"❌ Error initializing second opinion agent: {e}")
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

def analyze_medical_scan(image_path, scan_type_hint=None):
    """Analyze medical scan for abnormalities and provide detailed explanation."""
    agent = get_scan_analysis_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing medical scan for abnormalities and conditions..."):
            query = f"""
            Analyze this medical scan image thoroughly. 
            {f'The user indicated this is a {scan_type_hint} scan.' if scan_type_hint else ''}
            
            Provide a comprehensive analysis including:
            1. Scan type identification
            2. All abnormalities or findings
            3. Detailed explanation in simple language
            4. Urgency assessment
            5. What this means for the patient
            6. Recommended actions
            7. Areas of concern
            8. Follow-up requirements
            
            If you identify any emergency conditions, clearly mark them as such.
            Use compassionate, clear language that a patient can understand.
            """
            
            response = agent.run(query, images=[image_path])
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error analyzing scan: {e}")
        return None

def get_second_opinion(image_path, initial_analysis):
    """Get second opinion on the scan analysis."""
    agent = get_second_opinion_agent()
    if agent is None:
        return None

    try:
        with st.spinner("👨‍⚕️ Getting second opinion for verification..."):
            query = f"""
            Review this medical scan and the initial analysis below. Provide a second opinion focusing on:
            1. Accuracy of findings
            2. Additional observations
            3. Urgency level verification
            4. Any missed abnormalities
            
            Initial Analysis:
            {initial_analysis}
            
            Provide your professional second opinion and any additional insights.
            """
            
            response = agent.run(query, images=[image_path])
            return response.content.strip()
    except Exception as e:
        st.error(f"🚨 Error getting second opinion: {e}")
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

def determine_urgency_level(analysis_text):
    """Determine urgency level from analysis text."""
    analysis_lower = analysis_text.lower()
    
    if "emergency" in analysis_lower or "immediate" in analysis_lower:
        return "EMERGENCY", "🚨"
    elif "urgent" in analysis_lower:
        return "URGENT", "⚠️"
    elif "routine" in analysis_lower:
        return "ROUTINE", "📅"
    elif "normal" in analysis_lower and "no" in analysis_lower and "abnormal" in analysis_lower:
        return "NORMAL", "✅"
    else:
        # Check for emergency conditions
        for condition in EMERGENCY_CONDITIONS:
            if condition in analysis_lower:
                return "EMERGENCY", "🚨"
        return "ROUTINE", "📅"

def create_scan_report_pdf(image_data, analysis_results, second_opinion=None, scan_type=None):
    """Create a comprehensive PDF report of the scan analysis."""
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
            textColor=colors.navy
        )
        
        emergency_style = ParagraphStyle(
            'Emergency',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.red,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.pink,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.navy,
            spaceAfter=6
        )
        
        normal_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=12,
            leading=14
        )
        
        # Title
        content.append(Paragraph("🔬 Scan Reports Analyzer - Medical Scan Analysis Report", title_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Emergency warning if needed
        urgency_level, urgency_icon = determine_urgency_level(analysis_results)
        if urgency_level == "EMERGENCY":
            content.append(Paragraph(
                "🚨 EMERGENCY - SEEK IMMEDIATE MEDICAL ATTENTION",
                emergency_style
            ))
            content.append(Spacer(1, 0.25*inch))
        
        # Date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"📅 Report Generated: {current_datetime}", normal_style))
        if scan_type:
            content.append(Paragraph(f"🔍 Scan Type: {scan_type}", normal_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Disclaimer
        content.append(Paragraph(
            "⚠️ MEDICAL DISCLAIMER: This analysis is for informational purposes only and should not replace professional medical diagnosis. "
            "Always consult with a qualified healthcare professional for proper medical evaluation and treatment.",
            ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=10, textColor=colors.red,
                          borderWidth=1, borderColor=colors.red, borderPadding=5, backColor=colors.pink)
        ))
        content.append(Spacer(1, 0.25*inch))
        
        # Add scan image
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
                content.append(Paragraph("📸 Analyzed Medical Scan:", heading_style))
                content.append(img_obj)
                content.append(Spacer(1, 0.25*inch))
            except Exception as img_error:
                st.warning(f"Could not add image to PDF: {img_error}")
        
        # Analysis results
        content.append(Paragraph("🔬 Scan Analysis Results:", heading_style))
        
        # Format analysis results
        if analysis_results:
            clean_analysis = analysis_results.replace('<', '&lt;').replace('>', '&gt;')
            paragraphs = clean_analysis.split('\n')
            for para in paragraphs:
                if para.strip():
                    content.append(Paragraph(para.strip(), normal_style))
            content.append(Spacer(1, 0.25*inch))
        
        # Second opinion if available
        if second_opinion:
            content.append(Paragraph("👨‍⚕️ Second Opinion:", heading_style))
            clean_second_opinion = second_opinion.replace('<', '&lt;').replace('>', '&gt;')
            paragraphs = clean_second_opinion.split('\n')
            for para in paragraphs:
                if para.strip():
                    content.append(Paragraph(para.strip(), normal_style))
            content.append(Spacer(1, 0.25*inch))
        
        # Footer
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph("© 2025 Scan Reports Analyzer | Powered by Advanced AI Medical Analysis", 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)))
        
        pdf.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"📄 Error creating PDF report: {e}")
        return None

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'second_opinion' not in st.session_state:
        st.session_state.second_opinion = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'scan_type' not in st.session_state:
        st.session_state.scan_type = None
    if 'urgency_level' not in st.session_state:
        st.session_state.urgency_level = None

    # Header
    st.title("🔬 Scan Reports Analyzer")
    st.markdown("### Advanced Medical Scan Analysis & Abnormality Detection")
    
    # Medical disclaimer
    st.error("""
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
    
    This tool is designed to provide preliminary analysis of medical scans for educational purposes only. 
    It is NOT a replacement for professional medical diagnosis, consultation, or treatment. 
    
    **Always consult with a qualified healthcare professional for:**
    - Proper medical diagnosis and treatment
    - Interpretation of medical scans and test results
    - Medical emergencies and urgent health concerns
    - Any questions about your health condition
    
    **If you have a medical emergency, call emergency services immediately.**
    """)
    
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Medical Scan")
        
        # Scan type selection
        scan_type = st.selectbox(
            "What type of medical scan is this? (Optional)",
            ["Not Sure", "X-Ray", "CT Scan", "MRI", "Ultrasound", "Mammogram", 
             "Bone Scan", "PET Scan", "ECG/EKG", "Blood Test Report", "Other"],
            help="Selecting the correct scan type helps improve analysis accuracy"
        )
        
        uploaded_file = st.file_uploader(
            "Upload your medical scan image",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
            help="Upload a clear, high-quality image of your medical scan or report"
        )
        
        if uploaded_file:
            # Display uploaded image
            resized_image = resize_image_for_display(uploaded_file)
            if resized_image:
                st.image(resized_image, caption="Uploaded Medical Scan", width=MAX_IMAGE_WIDTH)
                
                # File info
                file_size = len(uploaded_file.getvalue()) / 1024
                st.info(f"**{uploaded_file.name}** • {file_size:.1f} KB")
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            analyze_button = st.button("🔬 Analyze Scan", type="primary")
        
        with col1b:
            second_opinion_button = st.button("👨‍⚕️ Get Second Opinion", 
                                            disabled=not st.session_state.analysis_results)
        
        # Process analysis
        if uploaded_file and analyze_button:
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                try:
                    scan_type_hint = scan_type if scan_type != "Not Sure" else None
                    analysis_result = analyze_medical_scan(temp_path, scan_type_hint)
                    
                    if analysis_result:
                        st.session_state.analysis_results = analysis_result
                        st.session_state.original_image = uploaded_file.getvalue()
                        st.session_state.scan_type = scan_type_hint
                        st.session_state.urgency_level = determine_urgency_level(analysis_result)
                        st.success("✅ Scan analysis completed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please try with a clearer image.")
                
                except Exception as e:
                    st.error(f"🚨 Analysis failed: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
        
        # Second opinion processing
        if uploaded_file and second_opinion_button and st.session_state.analysis_results:
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                try:
                    second_opinion_result = get_second_opinion(temp_path, st.session_state.analysis_results)
                    if second_opinion_result:
                        st.session_state.second_opinion = second_opinion_result
                        st.success("✅ Second opinion obtained successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Second opinion failed. Please try again.")
                except Exception as e:
                    st.error(f"🚨 Second opinion failed: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if st.session_state.analysis_results:
            # Display urgency level
            urgency_level, urgency_icon = st.session_state.urgency_level
            
            if urgency_level == "EMERGENCY":
                st.error(f"🚨 **EMERGENCY - SEEK IMMEDIATE MEDICAL ATTENTION**")
                st.error("Please visit the nearest emergency room or call emergency services immediately.")
            elif urgency_level == "URGENT":
                st.warning(f"⚠️ **URGENT - Medical consultation needed within 24-48 hours**")
            elif urgency_level == "ROUTINE":
                st.info(f"📅 **ROUTINE - Can be discussed at your next appointment**")
            else:
                st.success(f"✅ **NORMAL - No significant abnormalities detected**")
            
            st.markdown("---")
            
            # Display analysis results
            st.markdown("### 🔬 Detailed Analysis Report")
            st.markdown(st.session_state.analysis_results)
            
            # Display second opinion if available
            if st.session_state.second_opinion:
                st.markdown("---")
                st.markdown("### 👨‍⚕️ Second Opinion")
                st.markdown(st.session_state.second_opinion)
            
            # Download report
            st.markdown("---")
            st.subheader("📄 Download Complete Report")
            
            pdf_bytes = create_scan_report_pdf(
                st.session_state.original_image,
                st.session_state.analysis_results,
                st.session_state.second_opinion,
                st.session_state.scan_type
            )
            
            if pdf_bytes:
                download_filename = f"scan_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥 Download Complete Analysis Report",
                    data=pdf_bytes,
                    file_name=download_filename,
                    mime="application/pdf",
                    help="Download a comprehensive PDF report with all analysis results"
                )
        else:
            st.info("Upload a medical scan image and click 'Analyze Scan' to see detailed results here.")
    
    # Additional information section
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("🏥 Important Healthcare Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚨 When to Seek Emergency Care:**
            - Severe pain or discomfort
            - Difficulty breathing
            - Chest pain
            - Severe headache
            - Loss of consciousness
            - Sudden weakness or numbness
            """)
            
            st.markdown("""
            **📞 Emergency Contact Numbers:**
            - Emergency Services: 911 (US) / 112 (EU)
            - Poison Control: 1-800-222-1222 (US)
            - Mental Health Crisis: 988 (US)
            """)
        
        with col2:
            st.markdown("""
            **👨‍⚕️ Next Steps:**
            - Schedule appointment with your doctor
            - Bring this report to your consultation
            - Ask questions about your results
            - Follow recommended treatment plans
            - Request clarification if needed
            """)
            
            st.markdown("""
            **📋 What to Prepare for Doctor Visit:**
            - List of current medications
            - Medical history
            - Insurance information
            - Questions about your scan results
            - Any symptoms you're experiencing
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Scan Reports Analyzer | Advanced AI-Powered Medical Scan Analysis")
    st.markdown("*Powered by Gemini AI + Tavily Search for accurate medical information*")

if __name__ == "__main__":
    main()
