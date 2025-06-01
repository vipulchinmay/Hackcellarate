from flask import Flask, request, jsonify
import easyocr
import base64
from langchain_ollama import OllamaLLM
from io import BytesIO
from PIL import Image
from googletrans import Translator
import torch
import gc
import json
import os
from werkzeug.utils import secure_filename
import asyncio
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR Model with Multiple Languages
reader = easyocr.Reader(['en'], gpu=False)

# Initialize AI Model
model = OllamaLLM(model="seekhan", base_url="http://localhost:11434")

# Initialize Translator
translator = Translator()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def returnlang(selected_language):
    lang_dict = {
        "hi": "HINDI",
        "mr": "MARATHI",
        "bn": "BENGALI",
        "ta": "TAMIL",
        "te": "TELUGU",
        "kn": "KANNADA",
        "gu": "GUJARATI",
        "ml": "MALAYALAM",
        "pa": "PUNJABI",
        "en": "ENGLISH"
    }
    return lang_dict.get(selected_language, "ENGLISH")

def extract_medical_data(text):
    """Extract structured medical data from OCR text"""
    medical_data = {
        'test_results': [],
        'medications': [],
        'vital_signs': {},
        'dates': [],
        'doctor_notes': []
    }
    
    # Extract test results (looking for patterns like "Glucose: 120 mg/dl")
    test_patterns = [
        r'(?i)(glucose|sugar|hemoglobin|hb|cholesterol|bp|blood pressure|heart rate|temperature|weight|height|bmi)[\s:]+([0-9]+\.?[0-9]*)\s*([a-zA-Z/]+)?',
        r'(?i)(normal|high|low|elevated|decreased)[\s:]+([a-zA-Z\s]+)',
    ]
    
    for pattern in test_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            medical_data['test_results'].append({
                'parameter': match[0],
                'value': match[1] if len(match) > 1 else '',
                'unit': match[2] if len(match) > 2 else ''
            })
    
    # Extract dates
    date_patterns = [
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        r'\b(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})\b'
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        medical_data['dates'].extend(dates)
    
    return medical_data

def generate_health_analytics(user_data, medical_reports_data):
    """Generate comprehensive health analytics using AI"""
    
    # Prepare comprehensive prompt for health analysis
    prompt = f"""
    You are an advanced AI health analytics assistant. Analyze the following user profile and medical data to provide a comprehensive health report.

    USER PROFILE:
    - Name: {user_data.get('name', 'N/A')}
    - Age: {user_data.get('age', 'N/A')}
    - Gender: {user_data.get('gender', 'N/A')}
    - Blood Group: {user_data.get('blood_group', 'N/A')}
    - Medical Conditions: {user_data.get('medical_conditions', 'None reported')}
    - Date of Birth: {user_data.get('date_of_birth', 'N/A')}

    MEDICAL REPORTS DATA:
    {json.dumps(medical_reports_data, indent=2)}

    Please provide a comprehensive health analytics report in the following JSON format:

    {{
        "overall_health_score": <number between 1-100>,
        "health_status": "<Excellent/Good/Fair/Poor>",
        "key_findings": [
            "Finding 1",
            "Finding 2",
            "Finding 3"
        ],
        "risk_factors": [
            {{
                "factor": "Risk Factor Name",
                "level": "<Low/Medium/High>",
                "description": "Detailed description"
            }}
        ],
        "vital_signs_analysis": {{
            "blood_pressure": {{
                "status": "<Normal/High/Low>",
                "recommendation": "Specific advice"
            }},
            "heart_rate": {{
                "status": "<Normal/High/Low>",
                "recommendation": "Specific advice"
            }},
            "other_vitals": [
                {{
                    "parameter": "Parameter name",
                    "value": "Current value",
                    "status": "Normal/Abnormal",
                    "recommendation": "Advice"
                }}
            ]
        }},
        "dietary_recommendations": {{
            "foods_to_increase": [
                {{
                    "food": "Food name",
                    "reason": "Why this food is beneficial",
                    "serving_suggestion": "How much and when"
                }}
            ],
            "foods_to_avoid": [
                {{
                    "food": "Food name",
                    "reason": "Why to avoid",
                    "alternative": "Healthier alternative"
                }}
            ],
            "meal_plan_suggestions": [
                "Breakfast suggestion",
                "Lunch suggestion",
                "Dinner suggestion",
                "Snack suggestions"
            ]
        }},
        "exercise_recommendations": {{
            "cardio": {{
                "type": "Recommended cardio exercises",
                "frequency": "How often",
                "duration": "How long",
                "intensity": "Intensity level"
            }},
            "strength_training": {{
                "type": "Recommended strength exercises",
                "frequency": "How often",
                "sets_reps": "Sets and reps"
            }},
            "flexibility": {{
                "type": "Stretching/yoga recommendations",
                "frequency": "How often",
                "duration": "How long"
            }},
            "weekly_schedule": [
                "Monday: Activity",
                "Tuesday: Activity",
                "Wednesday: Activity",
                "Thursday: Activity",
                "Friday: Activity",
                "Saturday: Activity",
                "Sunday: Activity"
            ]
        }},
        "lifestyle_recommendations": [
            "Sleep recommendations",
            "Stress management",
            "Hydration advice",
            "Other lifestyle tips"
        ],
        "warning_signs": [
            "Symptom to watch for",
            "When to see a doctor"
        ],
        "follow_up_recommendations": {{
            "next_checkup": "When to schedule next checkup",
            "tests_needed": [
                "Test 1",
                "Test 2"
            ],
            "monitoring_parameters": [
                "What to monitor daily/weekly"
            ]
        }},
        "health_metrics": {{
            "cardiovascular_health": <score 1-10>,
            "metabolic_health": <score 1-10>,
            "immune_system": <score 1-10>,
            "mental_health": <score 1-10>,
            "nutritional_status": <score 1-10>
        }},
        "trends_analysis": [
            "Trend observation 1",
            "Trend observation 2"
        ],
        "personalized_tips": [
            "Tip 1 specific to user",
            "Tip 2 specific to user",
            "Tip 3 specific to user"
        ]
    }}

    Provide ONLY the JSON response without any additional text or formatting.
    """
    
    try:
        ai_response = model.invoke(input=prompt)
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Try to parse the JSON response
        try:
            analytics_data = json.loads(ai_response)
            return analytics_data
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic response
            return {
                "overall_health_score": 75,
                "health_status": "Good",
                "key_findings": ["Analysis completed based on available data"],
                "error": "Could not parse detailed analysis",
                "raw_response": ai_response
            }
            
    except Exception as e:
        print(f"Error in health analytics generation: {str(e)}")
        return {
            "error": f"Failed to generate analytics: {str(e)}",
            "overall_health_score": 0,
            "health_status": "Unknown"
        }

@app.route('/scan', methods=['POST'])
def scan_medicine():
    print("Medicine scanning endpoint called")
    try:
        print("✅ Image received. Starting the OCR processing...")

        data = request.json
        if not data or "image" not in data or "language" not in data:
            print("❌ No image received.")
            return jsonify({"error": "No image received"}), 400

        base64_str = data["image"]
        selected_language = data["language"]
        print("✅ Selected Language:", selected_language)

        # Convert Base64 to Image
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        
        output_lang = returnlang(selected_language)
        print("✅ Output Language:", output_lang)

        # OCR Processing
        print("🔍 Starting OCR...")
        ocr_result = reader.readtext(image, detail=0)
        extracted_text = " ".join(ocr_result)
        print("🔍 OCR Extracted text:", extracted_text)

        if not extracted_text:
            print("❌ No text found in the image.")
            return jsonify({"error": "No text found in the image"}), 400

        # AI Processing
        prompt = f"""You are a multilingual medical assistant. Your task is to extract key details from the given medicine label and respond ONLY in {output_lang}. 

        Extracted Text from the Image: {extracted_text}  

        Instructions for Extraction:
        From the above text, extract the following details:  
        1. Medicine Name - The official name of the medicine.  
        2. Description - A two-line explanation of the medicine's purpose.  
        3. Manufacturing Date (MFD) - The date when the medicine was manufactured.  
        4. Expiry Date (EXP) - The date after which the medicine should not be used.   

        IMPORTANT INSTRUCTIONS:
        - Respond strictly in {output_lang}.
        - DO NOT use English or any other language. 
        - Translate everything, including the structure of the response, into {output_lang}.
        """
        
        ai_response = model.invoke(input=prompt)
        torch.cuda.empty_cache()
        gc.collect()
        print("🤖 AI Response:", ai_response)

        if not ai_response:
            print("❌ AI model failed to extract valid details.")
            return jsonify({"error": "AI model failed to extract valid details"}), 500

        return jsonify({"raw_response": ai_response.strip()}), 200

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-medical-reports', methods=['POST'])
def analyze_medical_reports():
    """Analyze medical reports and generate health analytics"""
    print("Health analytics endpoint called")
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        user_data = data.get('userData', {})
        medical_reports_base64 = data.get('medicalReports', [])
        
        print(f"Processing health analytics for user: {user_data.get('name', 'Unknown')}")
        print(f"Number of medical reports: {len(medical_reports_base64)}")

        # Process medical reports with OCR
        all_medical_data = []
        
        for i, report_data in enumerate(medical_reports_base64):
            try:
                print(f"Processing medical report {i+1}...")
                
                # Decode base64 image
                image_bytes = base64.b64decode(report_data['base64'])
                image = Image.open(BytesIO(image_bytes))
                
                # OCR Processing
                ocr_result = reader.readtext(image, detail=0)
                extracted_text = " ".join(ocr_result)
                print(f"OCR extracted from report {i+1}: {extracted_text[:200]}...")
                
                # Extract structured medical data
                medical_data = extract_medical_data(extracted_text)
                medical_data['report_name'] = report_data.get('name', f'Report_{i+1}')
                medical_data['raw_text'] = extracted_text
                
                all_medical_data.append(medical_data)
                
            except Exception as e:
                print(f"Error processing report {i+1}: {str(e)}")
                continue

        # Generate comprehensive health analytics
        print("Generating health analytics...")
        health_analytics = generate_health_analytics(user_data, all_medical_data)
        
        # Add processing metadata
        health_analytics['analysis_date'] = datetime.now().isoformat()
        health_analytics['reports_processed'] = len(all_medical_data)
        health_analytics['user_id'] = user_data.get('_id', 'unknown')
        
        print("✅ Health analytics generated successfully")
        return jsonify({
            "success": True,
            "analytics": health_analytics,
            "medical_data_extracted": all_medical_data
        }), 200

    except Exception as e:
        print(f"❌ Error in health analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-analytics', methods=['POST'])
def save_analytics():
    """Save analytics data for future reference"""
    try:
        data = request.json
        user_id = data.get('user_id')
        analytics = data.get('analytics')
        
        # Create analytics directory if it doesn't exist
        analytics_dir = os.path.join(UPLOAD_FOLDER, 'analytics')
        os.makedirs(analytics_dir, exist_ok=True)
        
        # Save analytics to file
        filename = f"health_analytics_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(analytics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analytics, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": "Analytics saved successfully",
            "filename": filename
        }), 200
        
    except Exception as e:
        print(f"Error saving analytics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-analytics-history/<user_id>', methods=['GET'])
def get_analytics_history(user_id):
    """Get historical analytics for a user"""
    try:
        analytics_dir = os.path.join(UPLOAD_FOLDER, 'analytics')
        if not os.path.exists(analytics_dir):
            return jsonify({"analytics_history": []}), 200
        
        history = []
        for filename in os.listdir(analytics_dir):
            if filename.startswith(f'health_analytics_{user_id}_'):
                filepath = os.path.join(analytics_dir, filename)
                with open(filepath, 'r') as f:
                    analytics_data = json.load(f)
                    history.append({
                        "filename": filename,
                        "date": analytics_data.get('analysis_date'),
                        "overall_score": analytics_data.get('overall_health_score'),
                        "health_status": analytics_data.get('health_status')
                    })
        
        # Sort by date (newest first)
        history.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({"analytics_history": history}), 200
        
    except Exception as e:
        print(f"Error getting analytics history: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)