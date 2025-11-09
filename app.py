from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import os
from pathlib import Path
import time
import traceback
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('static/audio').mkdir(parents=True, exist_ok=True)

# Global variables for model
tokenizer = None
model = None
device = None

def load_model():
    """Load T5 model and tokenizer"""
    global tokenizer, model, device
    
    try:
        model_path = "./t5-small-cnn-finetuned"
        print("Loading model...")
        
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extract text"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload PDF, TXT, or DOCX files.'}), 400
        
        # Save file
        filename = f"{int(time.time())}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext == 'docx':
            text = extract_text_from_docx(file_path)
        elif file_ext == 'txt':
            text = extract_text_from_txt(file_path)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'}), 400
        
        # Clean up file
        os.remove(file_path)
        
        if not text or len(text) < 50:
            return jsonify({'success': False, 'error': 'File is empty or text is too short.'}), 400
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': file.filename
        })
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summary from text"""
    try:
        print("Summarize endpoint called")
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        text = data.get('text', '').strip()
        max_length_chars = data.get('max_length', 300)  # User selected character count
        
        if not text:
            print("No text provided")
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        if len(text) < 50:
            print("Text too short")
            return jsonify({'success': False, 'error': 'Text too short. Please provide at least 50 characters.'}), 400
        
        # Check if model is loaded
        if model is None or tokenizer is None:
            print("Model not loaded")
            return jsonify({'success': False, 'error': 'Model not loaded. Please restart the server.'}), 500
        
        print(f"Generating summary with max {max_length_chars} characters")
        
        # Convert characters to approximate tokens (1 token ≈ 4 characters)
        max_tokens = max(20, min(512, int(max_length_chars / 4)))
        min_tokens = max(10, int(max_tokens * 0.3))
        
        print(f"Token range: {min_tokens} - {max_tokens}")
        
        # Prepare input
        input_text = "summarize: " + text
        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids.to(device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_tokens,
                min_length=min_tokens,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Truncate to exact character limit if needed
        if len(summary) > max_length_chars:
            # Truncate at last complete sentence within limit
            truncated = summary[:max_length_chars]
            last_period = truncated.rfind('.')
            last_exclaim = truncated.rfind('!')
            last_question = truncated.rfind('?')
            last_sentence_end = max(last_period, last_exclaim, last_question)
            
            if last_sentence_end > max_length_chars * 0.7:  # If we found a sentence end in last 30%
                summary = truncated[:last_sentence_end + 1]
            else:
                summary = truncated.rstrip() + "..."
        
        print(f"Summary generated: {len(summary)} characters")
        
        # Calculate statistics
        original_words = len(text.split())
        original_chars = len(text)
        summary_words = len(summary.split())
        summary_chars = len(summary)
        compression_ratio = round((1 - summary_chars / original_chars) * 100, 1)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'stats': {
                'original_words': original_words,
                'original_chars': original_chars,
                'summary_words': summary_words,
                'summary_chars': summary_chars,
                'compression_ratio': compression_ratio
            }
        })
    
    except Exception as e:
        print(f"Error in summarize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate unique filename
        timestamp = int(time.time())
        audio_filename = f"summary_{timestamp}.mp3"
        audio_path = os.path.join('static', 'audio', audio_filename)
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        # Clean up old audio files (older than 1 hour)
        cleanup_old_files('static/audio', max_age_seconds=3600)
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/audio/{audio_filename}'
        })
    
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_old_files(directory, max_age_seconds=3600):
    """Remove files older than max_age_seconds"""
    try:
        current_time = time.time()
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)