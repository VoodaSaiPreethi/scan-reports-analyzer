import streamlit as st
import os
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.tavily import TavilyTools
from tempfile import NamedTemporaryFile
import datetime

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
    """Collect comprehensive patient information."""
    st.subheader("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        
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
        
        sleep = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])
        sleep_hours = st.number_input("Average Sleep Hours", min_value=3, max_value=12, value=7)
        stress = st.selectbox("Stress Level", ["Low", "Moderate", "High", "Very High"])
        
        occupation = st.text_input("Occupation", placeholder="Your job/profession")
        
        # Water intake
        water_intake = st.number_input("Daily Water Intake (glasses)", min_value=0, max_value=20, value=8)
    
    # Enhanced Diet Information
    st.subheader("🍽️ Detailed Diet Information")
    
    diet_col1, diet_col2 = st.columns(2)
    
    with diet_col1:
        diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Mediterranean", "Other"])
        
        meal_frequency = st.selectbox("Meal Frequency", ["2 meals/day", "3 meals/day", "4-5 small meals/day", "Irregular"])
        
        breakfast = st.text_area(
            "Typical Breakfast",
            placeholder="Describe what you usually eat for breakfast..."
        )
        
        lunch = st.text_area(
            "Typical Lunch",
            placeholder="Describe what you usually eat for lunch..."
        )
    
    with diet_col2:
        dinner = st.text_area(
            "Typical Dinner",
            placeholder="Describe what you usually eat for dinner..."
        )
        
        snacks = st.text_area(
            "Snacks & Beverages",
            placeholder="List your usual snacks, beverages, and their frequency..."
        )
        
        food_restrictions = st.text_area(
            "Food Restrictions/Preferences",
            placeholder="Any foods you avoid or prefer, cultural dietary restrictions..."
        )
        
        supplements = st.text_area(
            "Supplements/Vitamins",
            placeholder="List any supplements, vitamins, or health drinks you take..."
        )
    
    # Family History
    st.subheader("👥 Family Medical History")
    family_history = st.text_area(
        "Family Medical History",
        placeholder="List any significant medical conditions in your family (parents, siblings, grandparents)..."
    )
    
    # Recent Incidents
    st.subheader("⚠️ Recent Incidents & Symptoms")
    accidents = st.text_area(
        "Recent Accidents or Injuries",
        placeholder="Describe any recent accidents, injuries, or traumatic events..."
    )
    
    # Current Symptoms
    symptoms = st.text_area(
        "Current Symptoms",
        placeholder="Describe any current symptoms, pain, or discomfort you're experiencing..."
    )
    
    # Pain Scale
    pain_level = st.selectbox("Current Pain Level (0-10)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    return {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "previous_conditions": previous_conditions,
        "current_medications": current_medications,
        "allergies": allergies,
        "smoking": smoking,
        "alcohol": alcohol,
        "exercise": exercise,
        "sleep": sleep,
        "sleep_hours": sleep_hours,
        "stress": stress,
        "occupation": occupation,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "meal_frequency": meal_frequency,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "snacks": snacks,
        "food_restrictions": food_restrictions,
        "supplements": supplements,
        "family_history": family_history,
        "accidents": accidents,
        "symptoms": symptoms,
        "pain_level": pain_level
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
            - Age: {patient_info['age']}
            - Gender: {patient_info['gender']}
            - Height: {patient_info['height']} cm
            - Weight: {patient_info['weight']} kg
            - BMI: {bmi:.1f}
            - Previous Medical Conditions: {patient_info['previous_conditions']}
            - Current Medications: {patient_info['current_medications']}
            - Allergies: {patient_info['allergies']}
            - Smoking Status: {patient_info['smoking']}
            - Alcohol Consumption: {patient_info['alcohol']}
            - Exercise Frequency: {patient_info['exercise']}
            - Sleep Quality: {patient_info['sleep']} ({patient_info['sleep_hours']} hours/night)
            - Stress Level: {patient_info['stress']}
            - Occupation: {patient_info['occupation']}
            - Water Intake: {patient_info['water_intake']} glasses/day
            - Diet Type: {patient_info['diet_type']}
            - Meal Frequency: {patient_info['meal_frequency']}
            - Typical Breakfast: {patient_info['breakfast']}
            - Typical Lunch: {patient_info['lunch']}
            - Typical Dinner: {patient_info['dinner']}
            - Snacks & Beverages: {patient_info['snacks']}
            - Food Restrictions: {patient_info['food_restrictions']}
            - Supplements: {patient_info['supplements']}
            - Family Medical History: {patient_info['family_history']}
            - Recent Accidents/Injuries: {patient_info['accidents']}
            - Current Symptoms: {patient_info['symptoms']}
            - Pain Level: {patient_info['pain_level']}/10

            COMPREHENSIVE ANALYSIS REQUIREMENTS:
            1. STATE THE EXACT DISEASE OR PROBLEM in bold at the beginning
            2. Provide detailed explanation in simple layman's terms with analogies
            3. Assess severity and urgency with clear action timeline
            4. Recommend specific medical specialists with consultation urgency
            5. Create a personalized dietary plan based on the diagnosis
            6. Develop a complete lifestyle modification plan
            7. Correlate findings with patient's current lifestyle and diet
            8. Provide monitoring guidelines and red flags to watch for
            9. Consider the patient's occupation, stress level, and family history

            DIETARY PLAN REQUIREMENTS:
            - Specific foods to include and avoid
            - Meal timing and portion recommendations
            - Recipes or meal ideas if helpful
            - Supplements needed for recovery/management
            - Hydration recommendations

            LIFESTYLE PLAN REQUIREMENTS:
            - Exercise type, duration, and frequency
            - Sleep schedule optimization
            - Stress management techniques
            - Work-life balance recommendations
            - Daily routine modifications

            Please provide a thorough, empathetic analysis that considers all aspects of the patient's life.
            Use simple language and explain medical terms clearly.
            Format your response as plain text without section headers or markdown formatting.
            Be encouraging and supportive while being medically accurate.
            """
            
            response = agent.run(prompt, images=[image_path])
            return response.content
            
    except Exception as e:
        st.error(f"Error during analysis: {e}")
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

def create_report_for_download(patient_info, analysis_result):
    """Create a formatted report for download."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bmi = patient_info['weight'] / ((patient_info['height'] / 100) ** 2)
    
    report = f"""
COMPREHENSIVE MEDICAL SCAN ANALYSIS REPORT
Generated on: {current_date}

PATIENT INFORMATION:
Age: {patient_info['age']}
Gender: {patient_info['gender']}
Height: {patient_info['height']} cm
Weight: {patient_info['weight']} kg
BMI: {bmi:.1f}
Previous Medical Conditions: {patient_info['previous_conditions']}
Current Medications: {patient_info['current_medications']}
Allergies: {patient_info['allergies']}
Smoking Status: {patient_info['smoking']}
Alcohol Consumption: {patient_info['alcohol']}
Exercise Frequency: {patient_info['exercise']}
Sleep Quality: {patient_info['sleep']} ({patient_info['sleep_hours']} hours/night)
Stress Level: {patient_info['stress']}
Occupation: {patient_info['occupation']}
Water Intake: {patient_info['water_intake']} glasses/day

DIETARY INFORMATION:
Diet Type: {patient_info['diet_type']}
Meal Frequency: {patient_info['meal_frequency']}
Typical Breakfast: {patient_info['breakfast']}
Typical Lunch: {patient_info['lunch']}
Typical Dinner: {patient_info['dinner']}
Snacks & Beverages: {patient_info['snacks']}
Food Restrictions: {patient_info['food_restrictions']}
Supplements: {patient_info['supplements']}

MEDICAL HISTORY:
Family Medical History: {patient_info['family_history']}
Recent Accidents/Injuries: {patient_info['accidents']}
Current Symptoms: {patient_info['symptoms']}
Pain Level: {patient_info['pain_level']}/10

ANALYSIS RESULTS:
{analysis_result}

DISCLAIMER:
This AI analysis is for informational purposes only and should not replace professional medical consultation. Always consult with qualified healthcare professionals for proper diagnosis and treatment. The dietary and lifestyle recommendations should be reviewed with your healthcare provider before implementation.
"""
    return report

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
    st.markdown('<p class="subtitle">Upload your medical scan report for comprehensive AI-powered analysis with personalized diet and lifestyle recommendations</p>', unsafe_allow_html=True)
    
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
    
    # Always show patient information form
    st.markdown("---")
    patient_info = collect_patient_information()
    
    # Analysis button - only show if file is uploaded
    if uploaded_file and st.button("🔬 Analyze Medical Scan & Create Personalized Plan", type="primary"):
        if all([patient_info['age'], patient_info['gender'], patient_info['height'], patient_info['weight']]):
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                st.markdown("---")
                st.subheader("📊 Analysis Results & Personalized Recommendations")
                
                analysis_result = analyze_medical_scan(temp_path, patient_info)
                
                if analysis_result:
                    # Display analysis in a styled container
                    st.markdown(f'<div class="analysis-result">{analysis_result}</div>', unsafe_allow_html=True)
                    
                    # Create download button
                    report_content = create_report_for_download(patient_info, analysis_result)
                    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 Download Complete Analysis Report",
                        data=report_content,
                        file_name=f"comprehensive_medical_analysis_{current_date}.txt",
                        mime="text/plain",
                        help="Download the complete analysis report with personalized recommendations"
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
