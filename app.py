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
import time
import json

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
    /* [Previous CSS content remains exactly the same] */
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

# Enhanced system prompt and analysis instructions
# [Previous SYSTEM_PROMPT and MEDICAL_ANALYSIS_INSTRUCTIONS remain exactly the same]

@st.cache_resource
def get_medical_agent():
    """Initialize and cache the medical scan analysis agent with real-time capabilities."""
    try:
        return Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
            system_prompt=SYSTEM_PROMPT,
            instructions=MEDICAL_ANALYSIS_INSTRUCTIONS,
            tools=[TavilyTools(api_key=TAVILY_API_KEY)],
            markdown=True,
            show_tool_calls=True,
            debug_mode=False
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
    """Analyze medical scan with enhanced real-time data integration."""
    agent = get_medical_agent()
    if agent is None:
        return None

    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initial scan analysis
        status_text.text("🔬 Analyzing medical scan structure...")
        progress_bar.progress(10)
        
        # Step 2: Real-time medical data gathering
        status_text.text("🔍 Gathering real-time medical guidelines...")
        progress_bar.progress(30)
        
        # Step 3: Comprehensive analysis
        status_text.text("📊 Performing comprehensive medical analysis...")
        progress_bar.progress(50)
        
        # Enhanced query with strict formatting requirements
        enhanced_query = f"""
        MEDICAL SCAN ANALYSIS REQUEST - {scan_type}
        
        Patient Profile:
        - Age: {patient_data['age']} years
        - Gender: {patient_data['gender']}
        - Medical History: {patient_data['medical_history']}
        - Current Medications: {patient_data['medications']}
        - Current Symptoms: {patient_data['symptoms']}
        - Health Problems: {patient_data['health_problems']}
        - Diet: {patient_data['diet']}
        - Lifestyle: {patient_data['lifestyle']}
        - Habits: {patient_data['habits']}
        - Family History: {patient_data['family_history']}
        
        CRITICAL FORMATTING REQUIREMENTS:
        - You MUST format your response EXACTLY as specified in the instructions
        - Include ALL section headers exactly as written
        - If a section has no content, write "No significant findings" for that section
        - Never omit any section headers
        - Use the EXACT section headers provided in the instructions
        
        Please analyze this medical scan/report with the highest level of medical expertise and current knowledge,
        ensuring COMPLETE output with ALL sections included.
        """
        
        progress_bar.progress(70)
        status_text.text("🩺 Generating detailed medical interpretation...")
        
        # Run the enhanced analysis
        response = agent.run(enhanced_query, images=[image_path])
        
        progress_bar.progress(90)
        status_text.text("✅ Finalizing comprehensive analysis...")
        
        progress_bar.progress(100)
        status_text.text("✅ Medical analysis completed successfully!")
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return response.content.strip() if response else None
        
    except Exception as e:
        st.error(f"🚨 Error analyzing medical scan: {e}")
        return None

# [Previous save_uploaded_file and create_enhanced_medical_pdf functions remain the same]

def display_enhanced_analysis_section(title, content, icon, section_type="normal"):
    """Display analysis section with enhanced formatting and real-time context."""
    if content and content.lower() != "no significant findings":
        # Create appropriate styling based on section type
        if section_type == "emergency":
            st.markdown(f'<div class="emergency-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.error(content)
        elif section_type == "warning":
            st.markdown(f'<div class="warning-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.warning(content)
        elif section_type == "success":
            st.markdown(f'<div class="success-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.success(content)
        else:
            st.markdown(f'<div class="analysis-section">{icon} <strong>{title}</strong></div>', 
                       unsafe_allow_html=True)
            st.info(content)
        
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
    if 'analysis_timestamp' not in st.session_state:
        st.session_state.analysis_timestamp = None

    # [Previous header and patient information collection remains the same until results display]

    # Display Results - UPDATED SECTION
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Medical Analysis Results")
        st.markdown(f"**Analysis performed on:** {st.session_state.analysis_timestamp}")
        
        # Debug: Show raw analysis in expander
        with st.expander("🔍 View Raw Analysis (Debug)"):
            st.text(st.session_state.analysis_results)
        
        # Define sections with enhanced parsing
        sections = {
            "🔬 SCAN TYPE & PURPOSE": ("Scan Type & Purpose", "normal"),
            "📋 EXECUTIVE SUMMARY": ("Executive Summary", "normal"),
            "🔍 DETAILED FINDINGS": ("Detailed Findings", "normal"),
            "⚠️ ABNORMALITIES IDENTIFIED": ("Abnormalities Identified", "warning"),
            "🚨 EMERGENCY STATUS": ("Emergency Status", "emergency"),
            "📝 PROBLEM EXPLANATION": ("Problem Explanation", "normal"),
            "🔗 MEDICAL CORRELATION": ("Medical Correlation", "normal"),
            "👨‍⚕️ SPECIALIST CONSULTATIONS": ("Specialist Consultations", "normal"),
            "⚠️ IMMEDIATE PRECAUTIONS": ("Immediate Precautions", "warning"),
            "🌟 LIFESTYLE RECOMMENDATIONS": ("Lifestyle Recommendations", "success"),
            "📅 FOLLOW-UP REQUIREMENTS": ("Follow-up Requirements", "normal"),
            "❓ QUESTIONS FOR YOUR DOCTOR": ("Questions for Your Doctor", "normal"),
            "🚨 WARNING SIGNS": ("Warning Signs", "emergency"),
            "📊 PROGNOSIS & OUTLOOK": ("Prognosis & Outlook", "normal"),
            "🔬 LATEST RESEARCH": ("Latest Research", "normal")
        }
        
        # Check if we got a properly formatted response
        if not any(section in st.session_state.analysis_results for section in sections):
            st.warning("⚠️ The analysis format appears incomplete. Showing raw results:")
            st.markdown(st.session_state.analysis_results)
        else:
            # Process each section with robust parsing
            for section_marker, (section_title, section_type) in sections.items():
                # Enhanced pattern to catch more variations
                section_pattern = rf"{re.escape(section_marker)}[:\*]*\s*(.*?)(?=\n\s*[🔬📋🔍⚠️🚨📝🔗👨‍⚕️🌟📅❓📊]|$)"
                match = re.search(section_pattern, st.session_state.analysis_results, re.DOTALL | re.IGNORECASE)
                
                if match:
                    content = match.group(1).strip()
                    if content:
                        display_enhanced_analysis_section(section_title, content, "", section_type)
                    else:
                        st.markdown(f'<div class="analysis-section">🔍 <strong>{section_title}</strong></div>', 
                                  unsafe_allow_html=True)
                        st.info("No significant findings reported for this section.")
                        st.markdown("---")
        
        # [Rest of your code for emergency contact, PDF download, etc. remains the same]

if __name__ == "__main__":
    main()
