import streamlit as st
import os
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile

# API Keys
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    st.error("API keys are missing. Set the TAVILY_API_KEY and GOOGLE_API_KEY environment variables.")
    st.stop()

MAX_IMAGE_WIDTH = 400

# System prompt for medical analysis
SYSTEM_PROMPT = """You are a specialized medical AI assistant designed to analyze medical scan reports and provide comprehensive health assessments. Your role is to:

1. Analyze medical scans (X-rays, MRIs, CT scans, blood reports, etc.) and identify potential health issues
2. STATE THE EXACT DISEASE OR PROBLEM clearly and prominently
3. Provide detailed explanations in simple, layman-friendly language
4. Consider patient's medical history, lifestyle, and other factors for comprehensive analysis
5. Identify emergency situations and provide appropriate guidance
6. Always recommend consulting healthcare professionals for proper diagnosis and treatment

You must be thorough, accurate, and emphasize the importance of professional medical consultation."""

INSTRUCTIONS = """
When analyzing medical reports:

1. **PRIMARY ANALYSIS**: First, carefully examine the medical scan/report and STATE THE EXACT DISEASE OR PROBLEM in bold at the beginning of your response.

2. **DETAILED EXPLANATION**: Provide a comprehensive explanation of the findings in simple language that a layperson can understand.

3. **SEVERITY ASSESSMENT**: Clearly indicate if this is an emergency situation requiring immediate medical attention.

4. **LIFESTYLE CORRELATION**: Analyze how the patient's lifestyle, diet, habits, age, and medical history might be contributing to the condition.

5. **PRECAUTIONS & RECOMMENDATIONS**: Provide specific precautions and lifestyle modifications to prevent worsening of the condition.

6. **CONSULTATION GUIDANCE**: Recommend which type of specialist the patient should consult and urgency level.

7. **FOLLOW-UP**: Suggest monitoring and follow-up requirements.

Always emphasize that this is an AI analysis and professional medical consultation is essential for proper diagnosis and treatment.
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
    """Collect comprehensive patient information."""
    st.subheader("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Medical History
        st.subheader("🏥 Medical History")
        previous_conditions = st.text_area(
            "Previous Medical Conditions",
            placeholder="List any previous diagnoses, surgeries, or chronic conditions..."
        )
        
        current_medications = st.text_area(
            "Current Medications",
            placeholder="List all medications, supplements, and dosages you are currently taking..."
        )
        
        allergies = st.text_area(
            "Allergies",
            placeholder="List any known allergies to medications, foods, or other substances..."
        )
    
    with col2:
        # Lifestyle Information
        st.subheader("🏃‍♂️ Lifestyle Information")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasional", "Regular", "Heavy"])
        exercise = st.selectbox("Exercise Frequency", ["Never", "Rarely", "2-3 times/week", "Daily"])
        
        diet = st.text_area(
            "Diet Description",
            placeholder="Describe your typical diet, eating habits, and any dietary restrictions..."
        )
        
        sleep = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        stress = st.selectbox("Stress Level", ["Low", "Moderate", "High", "Very High"])
    
    # Family History
    st.subheader("👥 Family Medical History")
    family_history = st.text_area(
        "Family Medical History",
        placeholder="List any significant medical conditions in your family (parents, siblings, grandparents)..."
    )
    
    # Recent Incidents
    st.subheader("⚠️ Recent Incidents")
    accidents = st.text_area(
        "Recent Accidents or Injuries",
        placeholder="Describe any recent accidents, injuries, or traumatic events..."
    )
    
    # Current Symptoms
    st.subheader("🩺 Current Symptoms")
    symptoms = st.text_area(
        "Current Symptoms",
        placeholder="Describe any current symptoms, pain, or discomfort you're experiencing..."
    )
    
    return {
        "age": age,
        "gender": gender,
        "previous_conditions": previous_conditions,
        "current_medications": current_medications,
        "allergies": allergies,
        "smoking": smoking,
        "alcohol": alcohol,
        "exercise": exercise,
        "diet": diet,
        "sleep": sleep,
        "stress": stress,
        "family_history": family_history,
        "accidents": accidents,
        "symptoms": symptoms
    }

def analyze_medical_scan(image_path, patient_info):
    """Analyze the medical scan using AI agent with patient information."""
    agent = get_agent()
    if agent is None:
        return
    
    try:
        with st.spinner("🔬 Analyzing medical scan and patient information..."):
            # Create comprehensive prompt with patient information
            prompt = f"""
            Please analyze this medical scan report and provide a comprehensive health assessment.

            PATIENT INFORMATION:
            - Age: {patient_info['age']}
            - Gender: {patient_info['gender']}
            - Previous Medical Conditions: {patient_info['previous_conditions']}
            - Current Medications: {patient_info['current_medications']}
            - Allergies: {patient_info['allergies']}
            - Smoking Status: {patient_info['smoking']}
            - Alcohol Consumption: {patient_info['alcohol']}
            - Exercise Frequency: {patient_info['exercise']}
            - Diet: {patient_info['diet']}
            - Sleep Quality: {patient_info['sleep']}
            - Stress Level: {patient_info['stress']}
            - Family Medical History: {patient_info['family_history']}
            - Recent Accidents/Injuries: {patient_info['accidents']}
            - Current Symptoms: {patient_info['symptoms']}

            ANALYSIS REQUIREMENTS:
            1. **STATE THE EXACT DISEASE OR PROBLEM** (in bold at the beginning)
            2. Provide detailed explanation in layman's terms
            3. Assess severity and urgency
            4. Correlate findings with patient's lifestyle and history
            5. Recommend precautions and lifestyle modifications
            6. Specify which healthcare specialist to consult
            7. Indicate if this is an emergency situation
            8. Suggest follow-up timeline

            Please provide a thorough analysis considering all the patient information provided.
            """
            
            response = agent.run(prompt, images=[image_path])
            st.markdown(response.content)
            
    except Exception as e:
        st.error(f"Error during analysis: {e}")

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
        page_title="Medical Scan Report Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🩺"
    )
    
    # Custom CSS for light nude theme
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
        h1, h2, h3 {
            color: #5d4037;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🩺 Medical Scan Report Analyzer")
    st.markdown("### Upload your medical scan report for comprehensive AI-powered analysis")
    
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
                st.image(resized_image, caption="Uploaded Medical Scan", use_column_width=True)
        else:
            st.success("PDF file uploaded successfully")
        
        # Collect patient information
        patient_info = collect_patient_information()
        
        # Analysis button
        if st.button("🔬 Analyze Medical Scan Report", type="primary"):
            if all([patient_info['age'], patient_info['gender']]):
                temp_path = save_uploaded_file(uploaded_file)
                if temp_path:
                    st.markdown("---")
                    st.subheader("📊 Medical Analysis Report")
                    analyze_medical_scan(temp_path, patient_info)
                    os.unlink(temp_path)  # Clean up after analysis
            else:
                st.error("Please fill in at least your age and gender before analysis.")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Important Disclaimer:**
    This AI analysis is for informational purposes only and should not replace professional medical consultation. 
    Always consult with qualified healthcare professionals for proper diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()
