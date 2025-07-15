import streamlit as st
import os
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile
import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# API Keys
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    st.error("API keys are missing. Set the TAVILY_API_KEY and GOOGLE_API_KEY environment variables.")
    st.stop()

MAX_IMAGE_WIDTH = 400

# Enhanced system prompt for medical analysis
SYSTEM_PROMPT = """You are a specialized medical AI assistant designed to analyze medical scan reports and provide comprehensive health assessments. Your role is to:

1. Analyze medical scans (X-rays, MRIs, CT scans, blood reports, etc.) and identify potential health issues
2. STATE THE EXACT DISEASE OR PROBLEM clearly and prominently
3. Provide detailed explanations in simple, layman-friendly language that anyone can understand
4. Recommend specific medical specialists and urgency level
5. Create personalized lifestyle and dietary plans based on the diagnosis
6. Consider patient's medical history, lifestyle, and other factors for comprehensive analysis
7. Identify emergency situations and provide appropriate guidance
8. Always recommend consulting healthcare professionals for proper diagnosis and treatment

You must be thorough, accurate, empathetic, and emphasize the importance of professional medical consultation."""

INSTRUCTIONS = """
When analyzing medical reports, provide a comprehensive response with these sections:

1. **DIAGNOSIS IDENTIFICATION**: STATE THE EXACT DISEASE OR PROBLEM in bold at the beginning of your response.

2. **LAYMAN'S EXPLANATION**: Explain the condition in simple, everyday language that a person without medical background can easily understand. Use analogies and examples where helpful.

3. **SEVERITY & URGENCY ASSESSMENT**: Clearly indicate:
   - How serious is this condition?
   - Is this an emergency requiring immediate attention?
   - What happens if left untreated?

4. **MEDICAL SPECIALIST RECOMMENDATIONS**: Specify exactly which doctors to consult:
   - Primary specialist (e.g., Cardiologist, Orthopedist, Neurologist)
   - Secondary specialists if needed
   - When to see them (immediately, within a week, within a month)
   - What to expect during the consultation

5. **LIFESTYLE CORRELATION**: Analyze how the patient's current lifestyle factors might be contributing to or affecting the condition.

6. **PERSONALIZED DIETARY PLAN**: Based on the diagnosis, provide specific dietary recommendations:
   - Foods to include and why
   - Foods to avoid and why
   - Meal timing and frequency
   - Portion sizes if relevant
   - Specific nutrients needed for recovery/management

7. **LIFESTYLE MODIFICATION PLAN**: Create a detailed plan including:
   - Exercise recommendations (type, duration, frequency)
   - Sleep schedule adjustments
   - Stress management techniques
   - Daily routine modifications
   - Habits to develop or eliminate

8. **MONITORING & FOLLOW-UP**: Suggest what to monitor and when to follow up.

9. **PRECAUTIONS & RED FLAGS**: What symptoms to watch out for that would require immediate medical attention.

Always emphasize that this is an AI analysis and professional medical consultation is essential for proper diagnosis and treatment.
Provide your response in plain text format without any markdown headers or section titles.
Use clear, compassionate language that reduces anxiety while being informative.
"""

def resize_image_for_display(image_file):
    """Resize image for display only, returns bytes."""
    try:
        img = Image.open(image_file)
        image_file.seek(0)  # Reset file pointer
        aspect_ratio = img.height / img.width
        new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
        img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error resizing image: {e}")
        return None

@st.cache_resource  # Cache the agent
def get_agent():
    """Initialize and cache the AI agent."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=SYSTEM_PROMPT,
            instructions=INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
        )
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

def collect_patient_information():
    """Collect comprehensive patient information in a streamlined format."""
    st.subheader("📋 Patient Information")
    
    # Basic Information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    with col4:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    
    # Medical History (condensed)
    st.subheader("🏥 Medical Background")
    col1, col2 = st.columns(2)
    
    with col1:
        medical_history = st.text_area(
            "Medical History & Current Medications",
            placeholder="Previous conditions, current medications, allergies...",
            height=100
        )
        
        lifestyle_habits = st.text_area(
            "Lifestyle Habits",
            placeholder="Smoking, alcohol, exercise frequency, sleep pattern...",
            height=100
        )
    
    with col2:
        current_diet = st.text_area(
            "Current Diet Pattern",
            placeholder="Typical meals, diet type, food restrictions...",
            height=100
        )
        
        symptoms_concerns = st.text_area(
            "Current Symptoms & Concerns",
            placeholder="Current symptoms, pain level, recent incidents...",
            height=100
        )
    
    # Quick selections for common items
    st.subheader("⚡ Quick Assessment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High", "Very High"])
        exercise_freq = st.selectbox("Exercise Frequency", ["Never", "Rarely", "2-3 times/week", "Daily"])
    
    with col2:
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        pain_level = st.selectbox("Current Pain Level", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    with col3:
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol", ["Never", "Occasional", "Regular"])
    
    return {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "medical_history": medical_history,
        "lifestyle_habits": lifestyle_habits,
        "current_diet": current_diet,
        "symptoms_concerns": symptoms_concerns,
        "stress_level": stress_level,
        "exercise_freq": exercise_freq,
        "sleep_quality": sleep_quality,
        "pain_level": pain_level,
        "smoking": smoking,
        "alcohol": alcohol
    }

def analyze_medical_scan(image_path, patient_info):
    """Analyze the medical scan using AI agent with patient information."""
    agent = get_agent()
    if agent is None:
        return None
    
    try:
        with st.spinner("🔬 Analyzing medical scan and creating personalized health plan..."):
            # Calculate BMI
            bmi = patient_info['weight'] / ((patient_info['height'] / 100) ** 2)
            
            # Create comprehensive prompt with patient information
            prompt = f"""
            Please analyze this medical scan report and provide a comprehensive health assessment with personalized recommendations.

            PATIENT INFORMATION:
            - Age: {patient_info['age']} years
            - Gender: {patient_info['gender']}
            - Height: {patient_info['height']} cm
            - Weight: {patient_info['weight']} kg
            - BMI: {bmi:.1f}
            - Medical History & Medications: {patient_info['medical_history']}
            - Lifestyle Habits: {patient_info['lifestyle_habits']}
            - Current Diet Pattern: {patient_info['current_diet']}
            - Current Symptoms & Concerns: {patient_info['symptoms_concerns']}
            - Stress Level: {patient_info['stress_level']}
            - Exercise Frequency: {patient_info['exercise_freq']}
            - Sleep Quality: {patient_info['sleep_quality']}
            - Pain Level: {patient_info['pain_level']}/10
            - Smoking Status: {patient_info['smoking']}
            - Alcohol Consumption: {patient_info['alcohol']}

            COMPREHENSIVE ANALYSIS REQUIREMENTS:
            1. STATE THE EXACT DISEASE OR PROBLEM in bold at the beginning
            2. Provide detailed explanation in simple layman's terms with analogies
            3. Assess severity and urgency with clear action timeline
            4. Recommend specific medical specialists with consultation urgency
            5. Create a personalized dietary plan based on the diagnosis
            6. Develop a complete lifestyle modification plan
            7. Correlate findings with patient's current lifestyle and diet
            8. Provide monitoring guidelines and red flags to watch for
            9. Consider all patient factors for comprehensive care

            Please provide a thorough, empathetic analysis that considers all aspects of the patient's condition.
            Use simple language and explain medical terms clearly.
            Format your response as plain text without section headers or markdown formatting.
            Be encouraging and supportive while being medically accurate.
            """
            
            response = agent.run(prompt, images=[image_path])
            return response.content
            
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

def create_pdf_report(patient_info, analysis_result):
    """Create a formal PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2E5E8A')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.HexColor('#2E5E8A')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Build the PDF content
    story = []
    
    # Title
    story.append(Paragraph("COMPREHENSIVE MEDICAL SCAN ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Report Information
    current_date = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {current_date}", body_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Table
    bmi = patient_info['weight'] / ((patient_info['height'] / 100) ** 2)
    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    
    patient_data = [
        ['Patient Demographics', ''],
        ['Age', f"{patient_info['age']} years"],
        ['Gender', patient_info['gender']],
        ['Height', f"{patient_info['height']} cm"],
        ['Weight', f"{patient_info['weight']} kg"],
        ['BMI', f"{bmi:.1f} ({bmi_category})"],
        ['', ''],
        ['Health Assessment', ''],
        ['Stress Level', patient_info['stress_level']],
        ['Exercise Frequency', patient_info['exercise_freq']],
        ['Sleep Quality', patient_info['sleep_quality']],
        ['Pain Level', f"{patient_info['pain_level']}/10"],
        ['Smoking Status', patient_info['smoking']],
        ['Alcohol Consumption', patient_info['alcohol']]
    ]
    
    table = Table(patient_data, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F4FD')),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#E8F4FD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E5E8A')),
        ('TEXTCOLOR', (0, 7), (-1, 7), colors.HexColor('#2E5E8A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 7), (-1, 7), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Medical History Section
    if patient_info['medical_history'].strip():
        story.append(Paragraph("Medical History & Current Medications", heading_style))
        story.append(Paragraph(patient_info['medical_history'], body_style))
        story.append(Spacer(1, 12))
    
    # Lifestyle Information
    if patient_info['lifestyle_habits'].strip():
        story.append(Paragraph("Lifestyle Habits", heading_style))
        story.append(Paragraph(patient_info['lifestyle_habits'], body_style))
        story.append(Spacer(1, 12))
    
    # Current Diet
    if patient_info['current_diet'].strip():
        story.append(Paragraph("Current Diet Pattern", heading_style))
        story.append(Paragraph(patient_info['current_diet'], body_style))
        story.append(Spacer(1, 12))
    
    # Current Symptoms
    if patient_info['symptoms_concerns'].strip():
        story.append(Paragraph("Current Symptoms & Concerns", heading_style))
        story.append(Paragraph(patient_info['symptoms_concerns'], body_style))
        story.append(Spacer(1, 12))
    
    # Analysis Results
    story.append(Paragraph("COMPREHENSIVE ANALYSIS & RECOMMENDATIONS", heading_style))
    story.append(Spacer(1, 12))
    
    # Split analysis into paragraphs for better formatting
    analysis_paragraphs = analysis_result.split('\n\n')
    for paragraph in analysis_paragraphs:
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), body_style))
            story.append(Spacer(1, 8))
    
    # Disclaimer
    story.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        leading=12,
        textColor=colors.HexColor('#666666')
    )
    
    story.append(Paragraph("<b>MEDICAL DISCLAIMER</b>", heading_style))
    story.append(Paragraph(
        "This AI-generated analysis is for informational and educational purposes only and should not replace professional medical consultation. "
        "The recommendations provided are based on general medical knowledge and should be reviewed with qualified healthcare professionals "
        "before implementation. Always consult with your doctor, specialist, or certified healthcare provider for proper diagnosis, treatment, "
        "and personalized medical advice. In case of medical emergencies, seek immediate professional medical attention.",
        disclaimer_style
    ))
    
    # Footer
    story.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#888888')
    )
    story.append(Paragraph(f"Report generated by Medical Scan Analyzer | {current_date}", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk."""
    try:
        with NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return temp_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def main():
    # Page configuration with light theme
    st.set_page_config(
        page_title="Enhanced Medical Scan Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🩺"
    )
    
    # Custom CSS for light nude theme and centered heading
    st.markdown("""
    <style>
        .main {
            background-color: #faf8f6;
        }
        .stApp {
            background-color: #faf8f6;
        }
        .stButton>button {
            background-color: #e8d5c7;
            color: #5d4037;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #d7c4b7;
        }
        .stSelectbox>div>div {
            background-color: #f5f1ed;
        }
        .stTextArea>div>div>textarea {
            background-color: #f5f1ed;
        }
        .stNumberInput>div>div>input {
            background-color: #f5f1ed;
        }
        .stTextInput>div>div>input {
            background-color: #f5f1ed;
        }
        h1, h2, h3 {
            color: #5d4037;
        }
        .main-title {
            text-align: center;
            font-weight: bold;
            color: #5d4037;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #5d4037;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .analysis-result {
            background-color: #f5f1ed;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #e8d5c7;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-title">🩺 Enhanced Medical Scan Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload your medical scan report for comprehensive AI-powered analysis with personalized recommendations</p>', unsafe_allow_html=True)
    
    # File upload section
    st.subheader("📁 Upload Medical Scan Report")
    uploaded_file = st.file_uploader(
        "Upload medical scan report (X-ray, MRI, CT scan, blood report, etc.)",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Upload a clear image of your medical scan report for analysis"
    )
    
    if uploaded_file:
        # Display uploaded image
        if uploaded_file.type != "application/pdf":
            resized_image = resize_image_for_display(uploaded_file)
            if resized_image:
                st.image(resized_image, caption="Uploaded Medical Scan", use_container_width=True)
        else:
            st.success("PDF file uploaded successfully")
    
    # Always show patient information form
    st.markdown("---")
    patient_info = collect_patient_information()
    
    # Analysis button - only show if file is uploaded
    if uploaded_file and st.button("🔬 Analyze Medical Scan & Generate Report", type="primary"):
        if all([patient_info['age'], patient_info['gender'], patient_info['height'], patient_info['weight']]):
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                st.markdown("---")
                st.subheader("📊 Analysis Results & Personalized Recommendations")
                
                analysis_result = analyze_medical_scan(temp_path, patient_info)
                
                if analysis_result:
                    # Display analysis in a styled container
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)
                    
                    # Create PDF download button
                    pdf_buffer = create_pdf_report(patient_info, analysis_result)
                    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"Medical_Analysis_Report_{current_date}.pdf",
                        mime="application/pdf",
                        help="Download the complete analysis report as a professional PDF document"
                    )
                
                os.unlink(temp_path)  # Clean up after analysis
        else:
            st.error("Please fill in at least your age, gender, height, and weight before analysis.")
    
    # Show message if no file uploaded
    if not uploaded_file:
        st.info("Please upload a medical scan report to proceed with comprehensive analysis.")
    
    # Enhanced Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Important Medical Disclaimer:**
    
    This AI analysis is for informational and educational purposes only and should not replace professional medical consultation. 
    The recommendations provided are based on general medical knowledge and should be reviewed with qualified healthcare professionals 
    before implementation. Always consult with your doctor, specialist, or certified nutritionist for proper diagnosis, treatment, 
    and personalized dietary/lifestyle advice.
    
    **🚨 Emergency Situations:** If you experience severe symptoms, chest pain, difficulty breathing, or any medical emergency, 
    seek immediate medical attention by calling emergency services or visiting the nearest emergency room.
    """)

if __name__ == "__main__":
    main()
