from flask import Flask, request, jsonify
import easyocr
import base64
from langchain_ollama import OllamaLLM
from io import BytesIO
from PIL import Image
from googletrans import Translator  # Added translation support

app = Flask(__name__)

# Initialize OCR Model with Multiple Languages
reader = easyocr.Reader(['en'])

# Initialize AI Model
model = OllamaLLM(model="seekhan")

# Initialize Translator
translator = Translator()

# Function to return language name from code
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
    return lang_dict.get(selected_language, "ENGLISH")  # Default to English

@app.route('/scan', methods=['POST'])
def scan_medicine():
    print("this is running")
    try:
        print("✅ Image received. Starting the OCR processing...")

        # Get Image from Request
        data = request.json
        if not data or "image" not in data or "language" not in data:
            print("❌ No image received.")
            return jsonify({"error": "No image received"}), 400

        base64_str = data["image"]
        selected_language = data["language"]  # ✅ Get selected language from request
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

        # AI Processing with Seekhan
        print("🤖 Sending extracted text to AI model for processing...")
        prompt = f"""You are a multilingual medical assistant. Your task is to extract key details from the given medicine label and respond ONLY in {output_lang}. 

        ---
        Extracted Text from the Image:
        
        {extracted_text}  

        ---
        Instructions for Extraction:
        From the above text, extract the following details:  
        1. Medicine Name - The official name of the medicine.  
        2. Description - A two-line explanation of the medicine's purpose.  
        3. Manufacturing Date (MFD) - The date when the medicine was manufactured.  
        4. Expiry Date (EXP) - The date after which the medicine should not be used.   

        ---
        IMPORTANT INSTRUCTIONS:
        - Respond strictly in {output_lang}.
        - DO NOT use English or any other language. 
        - Translate everything, including the structure of the response, into {output_lang}.
        - Use this structured format, fully translated into {output_lang}:
        """
        print(prompt)
        print("sending to model")
        ai_response = model.invoke(input=prompt)
        print("🤖 AI Response:", ai_response)

        # If AI response is empty, return an error
        if not ai_response:
            print("❌ AI model failed to extract valid details.")
            return jsonify({"error": "AI model failed to extract valid details"}), 500

        return jsonify({"raw_response": ai_response.strip()}), 200

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)