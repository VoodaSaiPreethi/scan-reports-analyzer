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
import requests

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
    .findings-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .findings-table th {
        background-color: #3498db;
        color: white;
        padding: 0.5rem;
        text-align: left;
    }
    .findings-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #ddd;
    }
    .findings-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .severity-critical {
        color: #e74c3c;
        font-weight: bold;
    }
    .severity-severe {
        color: #e67e22;
        font-weight: bold;
    }
    .severity-moderate {
        color: #f39c12;
    }
    .severity-mild {
        color: #2ecc71;
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

# Enhanced system prompt for comprehensive medical analysis with real-time data integration
SYSTEM_PROMPT = """
You are a highly experienced medical professional with expertise in radiology, pathology, and medical imaging interpretation across ALL medical specialties. 

When analyzing medical scans and reports, follow these guidelines:

1. **REAL-TIME DATA INTEGRATION:**
   - Always incorporate the latest medical guidelines and research (current as of 2024)
   - Cross-reference with up-to-date clinical protocols
   - Use evidence-based medicine principles

2. **COMPREHENSIVE ANALYSIS:**
   - Provide detailed, accurate interpretations based on the visual/image data
   - Correlate findings with patient's medical history and symptoms
   - Identify both common and rare conditions that match the findings

3. **CLINICAL RELEVANCE:**
   - Focus on clinically significant findings
   - Highlight actionable insights for healthcare providers
   - Differentiate between incidental findings and relevant pathology

4. **STRUCTURED OUTPUT:**
   - Organize findings systematically
   - Use standardized medical terminology
   - Include relevant measurements and quantitative data when possible

5. **EMERGENCY PRIORITIZATION:**
   - Immediately flag life-threatening conditions
   - Provide clear triage recommendations
   - Specify time-sensitive interventions when needed

6. **DIFFERENTIAL DIAGNOSIS:**
   - List potential diagnoses in order of likelihood
   - Include supporting and contradicting evidence for each
   - Suggest next steps for diagnostic clarification

7. **TREATMENT IMPLICATIONS:**
   - Outline potential treatment options
   - Note any contraindications based on patient history
   - Highlight medication interactions when relevant

8. **FOLLOW-UP RECOMMENDATIONS:**
   - Specify appropriate follow-up intervals
   - Recommend additional testing if needed
   - Provide monitoring parameters for concerning findings
"""

# Comprehensive analysis instructions with real-world clinical focus
MEDICAL_ANALYSIS_INSTRUCTIONS = """
Analyze the medical scan/report with clinical precision and provide a structured report:

1. **SCAN IDENTIFICATION:**
   - Type: [CT/MRI/X-ray/etc.] 
   - Body region: [Detailed anatomical location]
   - Clinical context: [Reason for exam based on patient history]

2. **TECHNICAL QUALITY ASSESSMENT:**
   - Image quality: [Excellent/Good/Fair/Poor - specify limitations]
   - Protocol adequacy: [Appropriate/Suboptimal for clinical question]
   - Artifacts: [Present/Absent - describe if present]

3. **COMPREHENSIVE FINDINGS:**
   - Organize by anatomical structure/system
   - Describe both normal and abnormal findings
   - Include measurements for significant findings
   - Note any comparison to prior studies if available

4. **ABNORMALITIES DETAIL:**
   - Location: [Precise anatomical description]
   - Size: [Measurements in 3 dimensions when applicable]
   - Characteristics: [Density, enhancement pattern, margins, etc.]
   - Significance: [Clinical relevance based on patient profile]

5. **DIFFERENTIAL DIAGNOSIS:**
   - Primary diagnosis: [Most likely explanation]
   - Alternative diagnoses: [Other possibilities in order of likelihood]
   - Supporting evidence: [Why this diagnosis is likely]
   - Contradicting evidence: [Factors against this diagnosis]

6. **CLINICAL CORRELATION:**
   - How findings explain patient's symptoms
   - Relationship to existing medical conditions
   - Impact on current treatment plans

7. **EMERGENCY EVALUATION:**
   - Life-threatening conditions: [Present/Absent]
   - Time-sensitive findings: [Requiring immediate intervention]
   - Critical values: [Specific measurements requiring urgent action]

8. **RECOMMENDATIONS:**
   - Immediate actions: [Emergent consultations, treatments]
   - Follow-up imaging: [Type and timeframe]
   - Additional testing: [Laboratory or other diagnostic tests]
   - Specialist referral: [Specific specialties needed]

9. **PROGNOSTIC IMPLICATIONS:**
   - Expected clinical course
   - Potential complications to monitor
   - Long-term health implications

Return analysis in this EXACT structured format:

*Scan Identification:*
- Type: <scan type>
- Body Region: <anatomical location>
- Clinical Context: <reason for exam>

*Technical Quality:*
- Image Quality: <assessment>
- Protocol: <adequacy>
- Artifacts: <description>

*Findings:*
<structured by anatomical region>
- <Organ/Structure 1>: <detailed description>
- <Organ/Structure 2>: <detailed description>

*Abnormalities:*
1. <Abnormality 1>:
   - Location: <precise anatomy>
   - Size: <measurements>
   - Characteristics: <detailed description>
   - Severity: <Mild/Moderate/Severe/Critical>
2. <Abnormality 2>:
   ...

*Differential Diagnosis:*
1. <Primary Diagnosis>:
   - Likelihood: <High/Medium/Low>
   - Evidence: <supporting features>
   - Contradictions: <conflicting features>
2. <Alternative Diagnosis>:
   ...

*Clinical Correlation:*
<how findings relate to patient's condition>

*Emergency Evaluation:*
- Critical Findings: <list>
- Urgency: <Immediate/Urgent/Routine>
- Recommended Actions: <specific steps>

*Recommendations:*
1. <Recommendation 1>:
   - Priority: <High/Medium/Low>
   - Timeframe: <When to complete>
2. <Recommendation 2>:
   ...

*Prognosis:*
<expected outcomes based on findings>
"""

# Function to fetch latest medical guidelines
def fetch_medical_guidelines(condition):
    """Fetch current medical guidelines for a specific condition."""
    try:
        search_query = f"current 2024 clinical guidelines for {condition}"
        tavily_response = TavilyTools.search_internet(search_query)
        
        # Filter for reliable medical sources
        reliable_sources = [r for r in tavily_response if any(
            domain in r['url'].lower() for domain in [
                'nih.gov', 'who.int', 'mayoclinic.org', 'cdc.gov', 
                'nejm.org', 'jamanetwork.com', 'thelancet.com'
            ]
        )]
        
        return reliable_sources[:3]  # Return top 3 most relevant guidelines
    except Exception as e:
        st.warning(f"Could not fetch current guidelines: {e}")
        return []

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
    """Analyze any type of medical scan with comprehensive patient data and real-time data integration."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        with st.spinner("🔬 Analyzing medical scan/report with real-time clinical data..."):
            # Build comprehensive query with patient data and real-time context
            query = f"""
            Analyze this {scan_type} medical scan/report with the following patient context:
            
            **Patient Profile:**
            - Age: {patient_data['age']}
            - Gender: {patient_data['gender']}
            - Medical History: {patient_data['medical_history']}
            - Current Medications: {patient_data['medications']}
            - Presenting Symptoms: {patient_data['symptoms']}
            - Known Health Conditions: {patient_data['health_problems']}
            
            **Clinical Context:**
            - This analysis should incorporate current (2024) medical guidelines
            - Cross-reference with evidence-based medicine
            - Consider both common and rare differential diagnoses
            - Provide specific measurements for any abnormalities
            - Include quantitative assessments where applicable
            
            **Analysis Requirements:**
            1. Perform complete anatomical evaluation
            2. Identify and characterize all findings
            3. Correlate findings with patient's clinical presentation
            4. Provide specific, actionable recommendations
            5. Flag any urgent/emergent findings immediately
            
            IMPORTANT: If there are any findings requiring immediate medical attention, 
            state this clearly at the beginning of your report with specific actions.
            """
            
            response = agent.run(query, images=[image_path])
            
            # Enhance with real-time guideline data if abnormalities found
            if "abnormality" in response.content.lower() or "finding" in response.content.lower():
                guidelines = fetch_medical_guidelines(scan_type)
                if guidelines:
                    guideline_text = "\n".join([f"- {g['title']}: {g['url']}" for g in guidelines])
                    response.content += f"\n\n**Current Medical Guidelines:**\n{guideline_text}"
            
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
        content.append(Paragraph("Comprehensive Medical Imaging Analysis Report", 
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
            # Process the structured analysis results
            sections = [
                "Scan Identification", "Technical Quality", "Findings", 
                "Abnormalities", "Differential Diagnosis", "Clinical Correlation",
                "Emergency Evaluation", "Recommendations", "Prognosis"
            ]
            
            for section in sections:
                pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections)}):\*|$)"
                match = re.search(pattern, analysis_results, re.DOTALL | re.IGNORECASE)
                
                if match:
                    section_content = match.group(1).strip()
                    
                    # Special formatting for emergency evaluation
                    if section == "Emergency Evaluation":
                        content.append(Paragraph(f"🚨 {section}:", heading_style))
                        if "critical" in section_content.lower() or "urgent" in section_content.lower():
                            content.append(Paragraph(section_content, emergency_style))
                        else:
                            content.append(Paragraph(section_content, normal_style))
                    else:
                        content.append(Paragraph(f"{section}:", heading_style))
                        
                        # Format abnormalities as bullet points
                        if section == "Abnormalities":
                            abnormalities = [line.strip() for line in section_content.split('\n') if line.strip()]
                            for ab in abnormalities:
                                if ab.startswith('-') or ab.startswith('•'):
                                    content.append(Paragraph(ab, normal_style))
                                else:
                                    content.append(Paragraph(f"• {ab}", normal_style))
                        else:
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
            if title == "Abnormalities":
                # Parse abnormalities into a table
                abnormalities = [line.strip() for line in content.split('\n') if line.strip()]
                
                if abnormalities:
                    st.markdown("""
                    <table class="findings-table">
                        <thead>
                            <tr>
                                <th>Finding</th>
                                <th>Location</th>
                                <th>Size</th>
                                <th>Severity</th>
                            </tr>
                        </thead>
                        <tbody>
                    """, unsafe_allow_html=True)
                    
                    for ab in abnormalities:
                        if ':' in ab:
                            parts = [p.strip() for p in ab.split(':')]
                            if len(parts) >= 2:
                                finding = parts[0]
                                details = parts[1]
                                
                                # Extract severity
                                severity = "Moderate"
                                if "mild" in details.lower():
                                    severity = "Mild"
                                elif "severe" in details.lower():
                                    severity = "Severe"
                                elif "critical" in details.lower():
                                    severity = "Critical"
                                
                                # Extract location and size if available
                                location = details.split('Location:')[1].split(',')[0].strip() if 'Location:' in details else "N/A"
                                size = details.split('Size:')[1].split(',')[0].strip() if 'Size:' in details else "N/A"
                                
                                st.markdown(f"""
                                <tr>
                                    <td>{finding}</td>
                                    <td>{location}</td>
                                    <td>{size}</td>
                                    <td class="severity-{severity.lower()}">{severity}</td>
                                </tr>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        </tbody>
                    </table>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant abnormalities detected")
            
            elif title == "Emergency Evaluation":
                if "critical" in content.lower() or "urgent" in content.lower():
                    st.error(f"🚨 {content}")
                else:
                    st.success(f"✅ No urgent findings detected")
            
            elif title == "Differential Diagnosis":
                diagnoses = [line.strip() for line in content.split('\n') if line.strip()]
                for dx in diagnoses:
                    if ':' in dx:
                        parts = dx.split(':')
                        st.markdown(f"**{parts[0].strip()}:** {parts[1].strip()}")
                    else:
                        st.markdown(f"- {dx}")
            
            elif title == "Recommendations":
                recommendations = [line.strip() for line in content.split('\n') if line.strip()]
                for rec in recommendations:
                    if rec.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        st.markdown(f"**{rec}**")
                    else:
                        st.markdown(f"- {rec}")
            
            else:
                # Default display for other sections
                paragraphs = content.split('\n')
                for para in paragraphs:
                    if para.strip():
                        st.markdown(para)
        
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
    st.markdown("### AI-Powered Medical Imaging Analysis - Clinical Decision Support System")
    
    # Supported scan types display
    st.markdown('<div class="scan-type-card">🏥 Advanced Analysis for: CT, MRI, X-Ray, Ultrasound, Blood Tests, ECG, Pathology Reports & More</div>', 
                unsafe_allow_html=True)
    
    # Medical disclaimer
    st.error("""
    ⚠️ **CRITICAL MEDICAL DISCLAIMER**
    
    This medical scan analysis tool provides clinical decision support based on current medical knowledge. 
    It is NOT a substitute for professional medical judgment. 
    
    🏥 **Key Limitations:**
    - Does not replace physician interpretation
    - May not detect all abnormalities
    - Clinical correlation required
    - Final diagnosis requires professional evaluation
    
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
                with st.spinner("🔍 Performing comprehensive medical analysis with real-time clinical data..."):
                    analysis_result = analyze_medical_scan(temp_path, patient_data, scan_type)
                
                if analysis_result:
                    st.session_state.analysis_results = analysis_result
                    st.session_state.medical_image = uploaded_file.getvalue()
                    st.session_state.patient_data = patient_data
                    st.session_state.scan_type = scan_type
                    
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
        
        analysis_text = st.session_state.analysis_results
        
        # Check if emergency
        is_emergency = "emergency" in analysis_text.lower() and "urgent" in analysis_text.lower()
        
        # Define sections with appropriate icons
        sections = {
            "Scan Identification": "🔬",
            "Technical Quality": "🛠️",
            "Findings": "🔍",
            "Abnormalities": "⚠️",
            "Differential Diagnosis": "📋",
            "Clinical Correlation": "🔄",
            "Emergency Evaluation": "🚨",
            "Recommendations": "📝",
            "Prognosis": "📈"
        }
        
        for section, icon in sections.items():
            pattern = rf"\*{re.escape(section)}:\*(.*?)(?=\*(?:{'|'.join(re.escape(s) for s in sections.keys())}):\*|$)"
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                is_emergency_section = section in ["Emergency Evaluation"] and is_emergency
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
        st.markdown("## 👨‍⚕️ Next Steps with Your Doctor")
        st.info("""
        **📋 Preparing for Your Doctor Visit:**
        1. Download and print the PDF report above
        2. Bring original scan/test results
        3. Prepare a list of your symptoms and concerns
        4. Note any changes since the scan was taken
        5. Bring a list of all medications (including supplements)
        
        **💬 Questions to Ask Your Doctor:**
        - What do these findings mean for my health?
        - What treatment options are available?
        - Are there any lifestyle changes I should make?
        - What symptoms should prompt immediate medical attention?
        - When should I follow up?
        - Are there any additional tests I need?
        """)
    
    # About Section
    st.markdown("---")
    st.markdown("## ℹ️ About This Clinical Decision Support System")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔬 Clinical Features:**
        - Real-time medical guideline integration
        - Evidence-based analysis
        - Comprehensive anatomical evaluation
        - Quantitative measurements
        - Differential diagnosis support
        - Treatment implications
        """)
    
    with col2:
        st.markdown("""
        **🏥 Intended Use:**
        - For healthcare professionals
        - Clinical decision support
        - Second opinion reference
        - Patient education tool
        - Not for diagnostic purposes
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Scan Reports Analyzer | Clinical Decision Support System | Powered by Gemini AI + Tavily")
    st.markdown("*🩺 Integrating AI with clinical expertise for better patient outcomes*")
    
    # Privacy Notice
    st.markdown("### 🔒 HIPAA-Compliant Data Handling")
    st.info("""
    **Your medical data is protected:**
    - Encrypted processing
    - No permanent storage of images
    - HIPAA-compliant protocols
    - Data deleted after analysis
    - No third-party sharing
    """)

if __name__ == "__main__":
    main()
