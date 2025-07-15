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
import base64

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

@st.cache_resource
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
    """Collect patient information with an engaging, step-by-step approach."""
    
    # Initialize session state for form progress
    if 'form_step' not in st.session_state:
        st.session_state.form_step = 1
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    
    # Progress bar
    progress_value = (st.session_state.form_step - 1) / 4
    st.progress(progress_value)
    st.markdown(f"**Step {st.session_state.form_step} of 4**")
    
    # Step 1: Basic Information
    if st.session_state.form_step == 1:
        st.subheader("👤 Basic Information")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Your Age", 1, 120, 30)
            height = st.number_input("Height (cm)", 100, 250, 170)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            weight = st.number_input("Weight (kg)", 30, 300, 70)
        
        if st.button("Next →", key="step1_next"):
            st.session_state.patient_data.update({
                'age': age, 'gender': gender, 'height': height, 'weight': weight
            })
            st.session_state.form_step = 2
            st.rerun()
    
    # Step 2: Health Background (Simplified)
    elif st.session_state.form_step == 2:
        st.subheader("🏥 Health Background")
        
        # Quick checkboxes for common conditions
        st.markdown("**Do you have any of these conditions?** (Select all that apply)")
        conditions = st.multiselect(
            "",
            ["Diabetes", "High Blood Pressure", "Heart Disease", "Asthma", 
             "Arthritis", "Thyroid Issues", "Kidney Problems", "Liver Problems", "None of the above"],
            key="conditions"
        )
        
        # Medications
        medications = st.text_input(
            "Current Medications (if any)",
            placeholder="e.g., Metformin, Lisinopril, Vitamin D...",
            key="medications"
        )
        
        # Allergies
        allergies = st.text_input(
            "Known Allergies",
            placeholder="e.g., Penicillin, Peanuts, Shellfish...",
            key="allergies"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", key="step2_prev"):
                st.session_state.form_step = 1
                st.rerun()
        with col2:
            if st.button("Next →", key="step2_next"):
                st.session_state.patient_data.update({
                    'conditions': conditions, 'medications': medications, 'allergies': allergies
                })
                st.session_state.form_step = 3
                st.rerun()
    
    # Step 3: Lifestyle (Visual and Interactive)
    elif st.session_state.form_step == 3:
        st.subheader("🌟 Lifestyle Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visual sliders for lifestyle factors
            exercise_freq = st.select_slider(
                "Exercise Frequency",
                options=["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"],
                value="1-2 times/week"
            )
            
            sleep_hours = st.slider("Hours of Sleep per Night", 3, 12, 7)
            
            stress_level = st.select_slider(
                "Stress Level",
                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                value="Moderate"
            )
        
        with col2:
            # Quick lifestyle choices
            smoking = st.radio("Smoking Status", ["Never", "Former", "Current"], horizontal=True)
            alcohol = st.radio("Alcohol Consumption", ["Never", "Occasional", "Regular"], horizontal=True)
            
            # Diet type
            diet_type = st.selectbox(
                "Diet Type",
                ["Regular/Mixed", "Vegetarian", "Vegan", "Keto", "Mediterranean", "Low-carb", "Other"]
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", key="step3_prev"):
                st.session_state.form_step = 2
                st.rerun()
        with col2:
            if st.button("Next →", key="step3_next"):
                st.session_state.patient_data.update({
                    'exercise_freq': exercise_freq, 'sleep_hours': sleep_hours,
                    'stress_level': stress_level, 'smoking': smoking,
                    'alcohol': alcohol, 'diet_type': diet_type
                })
                st.session_state.form_step = 4
                st.rerun()
    
    # Step 4: Current Concerns (Final Step)
    elif st.session_state.form_step == 4:
        st.subheader("🎯 Current Concerns")
        
        # Pain assessment with visual
        pain_level = st.select_slider(
            "Current Pain Level",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=0,
            format_func=lambda x: f"{x}/10 {'😊' if x <= 3 else '😐' if x <= 6 else '😟'}"
        )
        
        # Symptoms checkboxes
        st.markdown("**Current Symptoms** (Select all that apply)")
        symptoms = st.multiselect(
            "",
            ["Fatigue", "Headaches", "Chest Pain", "Shortness of Breath", "Dizziness",
             "Nausea", "Back Pain", "Joint Pain", "Sleep Problems", "Appetite Changes",
             "Weight Changes", "Mood Changes", "None of the above"],
            key="symptoms"
        )
        
        # Additional concerns
        additional_concerns = st.text_area(
            "Any additional concerns or symptoms?",
            placeholder="Describe any other symptoms, concerns, or recent changes...",
            height=80
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", key="step4_prev"):
                st.session_state.form_step = 3
                st.rerun()
        with col2:
            if st.button("Complete Assessment ✓", key="step4_complete", type="primary"):
                st.session_state.patient_data.update({
                    'pain_level': pain_level, 'symptoms': symptoms,
                    'additional_concerns': additional_concerns
                })
                st.session_state.form_step = 5  # Completed
                st.rerun()
    
    # Return data only if form is complete
    if st.session_state.form_step == 5:
        return st.session_state.patient_data
    else:
        return None

def analyze_medical_scan(image_path, patient_info):
    """Analyze the medical scan using AI agent with patient information."""
    agent = get_agent()
    if agent is None:
        return None
    
    try:
        with st.spinner("🔬 Analyzing medical scan and creating personalized health plan..."):
            # Calculate BMI
            bmi = patient_info['weight'] / ((patient_info['height'] / 100) ** 2)
            
            # Format patient information nicely
            conditions_str = ", ".join(patient_info.get('conditions', [])) if patient_info.get('conditions') else "None reported"
            symptoms_str = ", ".join(patient_info.get('symptoms', [])) if patient_info.get('symptoms') else "None reported"
            
            # Create comprehensive prompt with patient information
            prompt = f"""
            Please analyze this medical scan report and provide a comprehensive health assessment with personalized recommendations.

            PATIENT INFORMATION:
            - Age: {patient_info['age']} years
            - Gender: {patient_info['gender']}
            - Height: {patient_info['height']} cm
            - Weight: {patient_info['weight']} kg
            - BMI: {bmi:.1f}
            - Existing Conditions: {conditions_str}
            - Current Medications: {patient_info.get('medications', 'None reported')}
            - Known Allergies: {patient_info.get('allergies', 'None reported')}
            - Exercise Frequency: {patient_info.get('exercise_freq', 'Not specified')}
            - Sleep Hours: {patient_info.get('sleep_hours', 'Not specified')} hours per night
            - Stress Level: {patient_info.get('stress_level', 'Not specified')}
            - Smoking Status: {patient_info.get('smoking', 'Not specified')}
            - Alcohol Consumption: {patient_info.get('alcohol', 'Not specified')}
            - Diet Type: {patient_info.get('diet_type', 'Not specified')}
            - Current Pain Level: {patient_info.get('pain_level', 0)}/10
            - Current Symptoms: {symptoms_str}
            - Additional Concerns: {patient_info.get('additional_concerns', 'None reported')}

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
    """Create a formal PDF report with proper error handling."""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
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
        
        # Format conditions and symptoms
        conditions_str = ", ".join(patient_info.get('conditions', [])) if patient_info.get('conditions') else "None reported"
        symptoms_str = ", ".join(patient_info.get('symptoms', [])) if patient_info.get('symptoms') else "None reported"
        
        patient_data = [
            ['Patient Demographics', ''],
            ['Age', f"{patient_info['age']} years"],
            ['Gender', patient_info['gender']],
            ['Height', f"{patient_info['height']} cm"],
            ['Weight', f"{patient_info['weight']} kg"],
            ['BMI', f"{bmi:.1f} ({bmi_category})"],
            ['', ''],
            ['Health Information', ''],
            ['Existing Conditions', conditions_str],
            ['Current Medications', patient_info.get('medications', 'None reported')],
            ['Known Allergies', patient_info.get('allergies', 'None reported')],
            ['Exercise Frequency', patient_info.get('exercise_freq', 'Not specified')],
            ['Sleep Hours per Night', f"{patient_info.get('sleep_hours', 'Not specified')}"],
            ['Stress Level', patient_info.get('stress_level', 'Not specified')],
            ['Smoking Status', patient_info.get('smoking', 'Not specified')],
            ['Alcohol Consumption', patient_info.get('alcohol', 'Not specified')],
            ['Diet Type', patient_info.get('diet_type', 'Not specified')],
            ['Current Pain Level', f"{patient_info.get('pain_level', 0)}/10"],
            ['Current Symptoms', symptoms_str]
        ]
        
        table = Table(patient_data, colWidths=[2.5*inch, 3.5*inch])
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
        
        # Additional Concerns Section
        if patient_info.get('additional_concerns', '').strip():
            story.append(Paragraph("Additional Concerns", heading_style))
            story.append(Paragraph(patient_info['additional_concerns'], body_style))
            story.append(Spacer(1, 12))
        
        # Analysis Results
        story.append(Paragraph("COMPREHENSIVE ANALYSIS & RECOMMENDATIONS", heading_style))
        story.append(Spacer(1, 12))
        
        # Split analysis into paragraphs for better formatting
        analysis_paragraphs = analysis_result.split('\n\n')
        for paragraph in analysis_paragraphs:
            if paragraph.strip():
                # Clean up any potential markup
                clean_paragraph = paragraph.strip().replace('**', '').replace('*', '')
                story.append(Paragraph(clean_paragraph, body_style))
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
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
        
    except Exception as e:
        st.error(f"Error creating PDF report: {e}")
        return None

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
    
    # Custom CSS for light nude theme and better UX
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
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #d7c4b7;
            transform: translateY(-2px);
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
        .stMultiSelect>div>div {
            background-color: #f5f1ed;
        }
        .stSlider>div>div>div>div {
            background-color: #e8d5c7;
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
        .step-container {
            background-color: #f5f1ed;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid #e8d5c7;
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
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(resized_image, caption="Uploaded Medical Scan", use_container_width=True)
        else:
            st.success("📄 PDF file uploaded successfully")
    
    # Patient information collection with improved UX
    st.markdown("---")
    st.subheader("📋 Patient Assessment")
    
    # Create a container for better styling
    with st.container():
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        patient_info = collect_patient_information()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis section - only show if both file and patient info are ready
    if uploaded_file and patient_info:
        st.markdown("---")
        
        # Show summary of collected information
        with st.expander("📊 Assessment Summary", expanded=False):
            bmi = patient_info['weight'] / ((patient_info['height'] / 100) ** 2)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age", f"{patient_info['age']} years")
                st.metric("BMI", f"{bmi:.1f}")
            
            with col2:
                st.metric("Exercise", patient_info.get('exercise_freq', 'Not specified'))
                st.metric("Sleep", f"{patient_info.get('sleep_hours', 'Not specified')} hours")
            
            with col3:
                st.metric("Stress Level", patient_info.get('stress_level', 'Not specified'))
                st.metric("Pain Level", f"{patient_info.get('pain_level', 0)}/10")
        
        # Analysis button
        if st.button("🔬 Analyze Medical Scan & Generate Report", type="primary", use_container_width=True):
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                st.markdown("---")
                st.subheader("📊 Analysis Results & Personalized Recommendations")
                
                analysis_result = analyze_medical_scan(temp_path, patient_info)
                
                if analysis_result:
                    # Display analysis in a styled container
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)
                    
                    # Create PDF and provide download
                    with st.spinner("📄 Generating PDF report..."):
                        pdf_data = create_pdf_report(patient_info, analysis_result)
                        
                        if pdf_data:
                            current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            st.success("✅ Report generated successfully!")
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_data,
                                file_name=f"Medical_Analysis_Report_{current_date}.pdf",
                                mime="application/pdf",
                                help="Download the complete analysis report as a professional PDF document",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to generate PDF report. Please try again.")
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Show appropriate messages based on current state
    elif not uploaded_file and not patient_info:
        st.info("📥 Please upload a medical scan report to begin the analysis process.")
    elif uploaded_file and not patient_info:
        st.info("📝 Please complete the patient assessment above to proceed with analysis.")
    elif not uploaded_file and patient_info:
        st.info("📥 Please upload a medical scan report to complete the analysis.")
    
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
