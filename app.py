from flask import Flask, request, render_template, send_file, session, redirect, url_for, flash, jsonify, send_from_directory
from flask_session import Session
import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sentence_transformers import SentenceTransformer, util
import pickle
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import concurrent.futures
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
from openpyxl import load_workbook
import yaml

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configuration Class
class AppConfig:
    def __init__(self, upload_folder, cache_dir, model_type, similarity_threshold, max_results, use_gpu):
        self.upload_folder = upload_folder
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.use_gpu = use_gpu

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        config_dict.setdefault('upload_folder', 'uploads')
        config_dict.setdefault('cache_dir', 'cache')
        config_dict.setdefault('model_type', 'resnet50')
        config_dict.setdefault('similarity_threshold', 0.1)
        config_dict.setdefault('max_results', 4000)
        config_dict.setdefault('use_gpu', False)
        return cls(**config_dict)

config = AppConfig.from_yaml('config.yaml')
app.config['UPLOAD_FOLDER'] = config.upload_folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

# Retry Strategy for HTTP Requests
retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session = requests.Session()
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)

# Models
image_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global Variables
df = pd.DataFrame(columns=['DBID', 'Product ID', 'Description', 'Fabric Type', 'Color', 'Category', 'Image_URL', 'Buyer Name', 'QTY (MOQ)', 'Season'])
df_filtered = df.copy()
image_urls = []
descriptions = []
fabric_types = []
colors = []
patterns = []
image_features = []
text_features = []
buyer_names = []
qty_moq = []
seasons = []
cache_file = os.path.join(config.cache_dir, 'image_features_multimodal_v2.pkl')

# Database Initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        garment_type TEXT,
        result_count INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    c.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                  ('admin', generate_password_hash('admin123', method='pbkdf2:sha256'), 'admin'))
    conn.commit()
    conn.close()

# Create Excel Template
def create_excel_template():
    template_data = {
        'DBID': ['Leave blank (auto-generated)'],
        'Product ID': ['PROD001'],
        'Description': ['Floral summer dress'],
        'Fabric Type': ['Cotton'],
        'Color': ['Red'],
        'Category': ['Dress'],
        'Buyer Name': ['John Doe'],
        'Image': ['Insert image here (right-click cell > Insert Picture)']
    }
    template_df = pd.DataFrame(template_data)
    template_path = 'garment_upload_template.xlsx'
    template_df.to_excel(template_path, index=False)

create_excel_template()
init_db()

# Load Database
def load_database():
    global df, df_filtered, image_urls, descriptions, fabric_types, colors, patterns, buyer_names, qty_moq, seasons
    if os.path.exists('new_garments_with_urls.xlsx'):
        df = pd.read_excel('new_garments_with_urls.xlsx')
        df = df.fillna('')
        column_mapping = {'uyer Nam': 'Buyer Name'}
        df = df.rename(columns=column_mapping)
        if 'QTY (MOQ)Description' in df.columns:
            df[['QTY (MOQ)', 'Description']] = df['QTY (MOQ)Description'].str.extract(r'(\d+)\s*(.*)')
            df = df.drop(columns=['QTY (MOQ)Description'])
            df['QTY (MOQ)'] = pd.to_numeric(df['QTY (MOQ)'], errors='coerce').fillna('')
            df['Description'] = df['Description'].fillna('')
        df_filtered = df.copy()
        image_urls = df.get('Image_URL', pd.Series([''] * len(df))).tolist()
        descriptions = df.get('Description', pd.Series([''] * len(df))).tolist()
        fabric_types = df.get('Fabric Type', pd.Series([''] * len(df))).tolist()
        colors = df.get('Color', pd.Series([''] * len(df))).tolist()
        patterns = df.get('Pattern', pd.Series([''] * len(df))).tolist()
        buyer_names = df.get('Buyer Name', pd.Series([''] * len(df))).tolist()
        qty_moq = df.get('QTY (MOQ)', pd.Series([''] * len(df))).tolist()
        seasons = df.get('Season', pd.Series([''] * len(df))).tolist()
        logging.debug(f"Loaded {len(df)} garments from database")

load_database()

# Feature Extraction
def load_image_with_timeout(url, timeout=10):
    try:
        response = http_session.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.content
        logging.warning(f"Failed to load image from {url}, status: {response.status_code}")
        return None
    except Exception as e:
        logging.error(f"Error loading image from {url}: {str(e)}")
        return None

def extract_image_features(image_path):
    try:
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Failed to load image: {image_path}")
            return None
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(np.expand_dims(img, axis=0))
        features = image_model.predict(img)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error extracting features from {image_path}: {str(e)}")
        return None

def extract_image_features_from_url(image_url):
    try:
        image_data = load_image_with_timeout(image_url)
        if image_data is None:
            return None
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        features = image_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error extracting features from URL {image_url}: {str(e)}")
        return None

def extract_text_features(description, fabric_type, color, pattern='', buyer_name='', qty_moq='', season=''):
    combined_text = f"{description} {fabric_type} {color} {pattern} {buyer_name} {qty_moq} {season}".lower()
    return text_model.encode(combined_text)

def batch_process_images(urls, batch_size=50):
    features = []
    total_batches = (len(urls) + batch_size - 1) // batch_size
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        logging.debug(f"Processing batch {i//batch_size + 1}/{total_batches}, URLs: {len(batch_urls)}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            batch_features = list(executor.map(extract_image_features_from_url, batch_urls))
        features.extend([f for f in batch_features if f is not None])
        time.sleep(1)  # Respect rate limits
    logging.debug(f"Completed feature extraction for {len(features)} images")
    return features

def load_or_compute_features():
    global image_features, text_features
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            image_features, text_features = pickle.load(f)
        logging.debug(f"Loaded cached features: {len(image_features)} image features, {len(text_features)} text features")
    else:
        logging.debug(f"Computing features for {len(image_urls)} images")
        image_features = batch_process_images(image_urls)
        text_features = [extract_text_features(desc, fab, col, pat, buy, qty, seas) 
                        for desc, fab, col, pat, buy, qty, seas in zip(descriptions, fabric_types, colors, patterns, buyer_names, qty_moq, seasons)]
        with open(cache_file, 'wb') as f:
            pickle.dump((image_features, text_features), f)
        logging.debug(f"Computed and cached {len(image_features)} image features, {len(text_features)} text features")

load_or_compute_features()

# Attribute Detection
def detect_dominant_color(img_array):
    img_hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    hue, saturation, value = mean_hsv
    if value < 100 and saturation > 50:
        return "dark"
    elif value > 200 and saturation > 50:
        return "light"
    else:
        return "medium"

def detect_category(img_array):
    height, width, _ = img_array.shape
    if height > width * 1.5:  # Rough heuristic for long garments
        return "Dress"
    return "Upper Wear"

def detect_description_features(img_array):
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if cv2.countNonZero(edges) > 1000:
        return ["lace", "embroidery"]
    return []

# Enhanced Matching Function
def find_similar_garments(new_image_path, garment_type="dress", threshold=config.similarity_threshold):
    new_img_features = extract_image_features(new_image_path)
    if new_img_features is None:
        logging.warning("No features extracted from uploaded image")
        return []
    
    # Detect attributes from uploaded image
    img = cv2.imread(new_image_path)
    img = cv2.resize(img, (224, 224))
    img_array = preprocess_input(np.expand_dims(img, axis=0))
    uploaded_color = detect_dominant_color(img_array[0])
    uploaded_category = detect_category(img_array[0])
    uploaded_descriptions = detect_description_features(img_array[0])
    text_query = f"{uploaded_color} {uploaded_category} {' '.join(uploaded_descriptions)}"
    new_text_features = text_model.encode(text_query)

    # Handle cases where features might be empty or None
    valid_img_features = [feat for feat in image_features if feat is not None and feat.size > 0]
    valid_text_features = [feat for feat in text_features if feat is not None and feat.size > 0]
    logging.debug(f"Number of valid image features: {len(valid_img_features)}, valid text features: {len(valid_text_features)}")
    if not valid_img_features or not valid_text_features:
        logging.warning("No valid features in database for comparison")
        return []

    # Compute similarities
    img_similarities = [util.cos_sim(new_img_features, feat_img).item() for feat_img in valid_img_features]
    text_similarities = [util.cos_sim(new_text_features, feat_text).item() for feat_text in valid_text_features]
    combined_scores = [0.5 * img_sim + 0.5 * text_sim for img_sim, text_sim in zip(img_similarities, text_similarities)]
    logging.debug(f"Min combined score: {min(combined_scores)}, Max combined score: {max(combined_scores)}")

    # Apply attribute-based scoring
    final_scores = []
    for idx, (img_sim, text_sim) in enumerate(zip(img_similarities, text_similarities)):
        db_color = colors[idx].lower() if idx < len(colors) else ''
        db_category = df_filtered.iloc[idx]['Category'].lower() if idx < len(df_filtered) else ''
        db_desc = descriptions[idx].lower() if idx < len(descriptions) else ''
        
        # More flexible attribute scoring
        color_similarity = 1.0 if uploaded_color in db_color or (uploaded_color == "dark" and any(c in db_color for c in ["black", "navy", "dark"])) else 1.2
        category_similarity = 1.0 if uploaded_category.lower() == db_category else 1.3
        desc_similarity = 1.0 if any(desc in db_desc for desc in uploaded_descriptions) else 1.1
        attribute_score = (color_similarity + category_similarity + desc_similarity) / 3
        
        combined_score = (0.5 * img_sim + 0.5 * text_sim) * attribute_score
        final_scores.append((combined_score, idx))

    # Sort and filter by threshold, then limit to max_results
    final_scores.sort(reverse=True)
    filtered_results = [(image_urls[idx], df_filtered.iloc[idx].to_dict()) 
                       for score, idx in final_scores if score >= threshold][:config.max_results]

    logging.debug(f"Found {len(filtered_results)} similar garments after threshold {threshold}")
    return filtered_results

def find_garments_by_text(query, category="", fabric=""):
    global df, text_features
    logging.debug(f"Received search: query='{query}', category='{category}', fabric='{fabric}'")
    
    # Normalize and split query into keywords
    query = query.lower().strip()
    keywords = [kw.strip() for kw in query.split() if kw.strip()]
    logging.debug(f"Extracted keywords: {keywords}")
    
    # Prepare combined text data from dataframe
    text_data = []
    for idx, row in df.iterrows():
        combined_text = (
            f"{row.get('Description', '')} "
            f"{row.get('Fabric Type', '')} "
            f"{row.get('Color', '')} "
            f"{row.get('Category', '')} "
            f"{row.get('Buyer Name', '') if 'Buyer Name' in df.columns else ''} "
            f"{row.get('QTY (MOQ)', '')} "
            f"{row.get('Season', '')}"
        ).lower()
        text_data.append((idx, combined_text))
    
    # Filter based on keywords (minimum 2 matches)
    matched_indices = []
    for idx, text in text_data:
        # Count matches across key fields
        matches = sum(
            any(kw in field.lower() for field in [
                df.at[idx, 'Color'],
                df.at[idx, 'Description'],
                df.at[idx, 'Category']
            ])
            for kw in keywords
        )
        
        # Require at least 2 keyword matches
        if matches >= 2:
            # Additional filters for category and fabric if provided
            cat_match = not category or category.lower() in df.at[idx, 'Category'].lower()
            fab_match = not fabric or fabric.lower() in df.at[idx, 'Fabric Type'].lower()
            
            if cat_match and fab_match:
                matched_indices.append(idx)
    
    logging.debug(f"Matched indices: {matched_indices}")
    
    # Prepare results
    results = []
    for idx in matched_indices:
        url = df.at[idx, 'Image_URL'] if pd.notna(df.at[idx, 'Image_URL']) else ''
        info = {
            'DBID': idx,
            'Product ID': df.at[idx, 'Product ID'],
            'Description': df.at[idx, 'Description'],
            'Fabric Type': df.at[idx, 'Fabric Type'],
            'Color': df.at[idx, 'Color'],
            'Category': df.at[idx, 'Category'],
            'Buyer Name': df.at[idx, 'Buyer Name'] if 'Buyer Name' in df.columns else '',
            'QTY (MOQ)': df.at[idx, 'QTY (MOQ)'],
            'Season': df.at[idx, 'Season']
        }
        results.append((url, info))
    
    logging.debug(f"Found {len(results)} garments for query: {query}")
    return results[:config.max_results]  # Limit to max_results from config
# History Functions
def save_to_history(user_id, garment_type, result_count):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO history (user_id, date, garment_type, result_count) VALUES (?, ?, ?, ?)',
              (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), garment_type, result_count))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT date, garment_type, result_count FROM history WHERE user_id = ? ORDER BY date DESC', (user_id,))
    history = c.fetchall()
    conn.close()
    return history

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            session['selected_items'] = []
            session.modified = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            password_hash = generate_password_hash(password, method='pbkdf2:sha256')
            c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                      (username, password_hash, role))
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session.get('username', 'User')
    role = session.get('role', 'user')
    session.modified = True
    return render_template('home.html', username=username, role=role)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        category = request.form.get('category', '').strip()
        fabric = request.form.get('fabric', '').strip()
        logging.debug(f"Received search: query='{query}', category='{category}', fabric='{fabric}'")
        if query or category or fabric:
            results = find_garments_by_text(query, category=category, fabric=fabric)
            logging.debug(f"Search returned {len(results)} results: {results}")
            session['search_results'] = results
            session['selected_items'] = []
            save_to_history(session['user_id'], 'text_search', len(results))
            return redirect(url_for('search_results'))
        flash('Please enter a search query or select filters')
        logging.debug("No valid search criteria provided, flashing message")
    return render_template('search.html')

@app.route('/search_results')
def search_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if 'selected_items' not in session:
        session['selected_items'] = []
    results = session.get('search_results', [])
    logging.debug(f"Rendering search_results with {len(results)} items")
    return render_template('search_results.html', results=results, selected_items=session['selected_items'])

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('upload_file'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('upload_file'))
        garment_type = request.form.get('garment_type', 'dress')
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('temp_upload.jpg'))
        file.save(temp_path)
        results = find_similar_garments(temp_path, garment_type=garment_type)
        logging.debug(f"Results from find_similar_garments: {results}")
        save_to_history(session['user_id'], garment_type, len(results))
        os.remove(temp_path)
        session['results'] = results
        session['selected_items'] = []
        return redirect(url_for('results'))
    return render_template('upload.html')

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if 'selected_items' not in session:
        session['selected_items'] = []
    results = session.get('results', [])
    logging.debug(f"Results passed to template: {results}")
    return render_template('results.html', results=results, selected_items=session['selected_items'])

@app.route('/update_selection', methods=['POST'])
def update_selection():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    dbid = data.get('dbid')
    selected = data.get('selected')
    if 'selected_items' not in session:
        session['selected_items'] = []
    if selected and dbid not in session['selected_items']:
        session['selected_items'].append(str(dbid))
    elif not selected and dbid in session['selected_items']:
        session['selected_items'].remove(str(dbid))
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    history = get_user_history(session['user_id'])
    return render_template('history.html', history=history)

@app.route('/admin')
def admin_panel():
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    return render_template('admin.html', garments=df)

@app.route('/add_garment', methods=['POST'])
def add_garment():
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    global df, df_filtered, image_urls, descriptions, fabric_types, colors, patterns, image_features, text_features, buyer_names, qty_moq, seasons
    product_id = request.form['product_id']
    description = request.form['description']
    fabric_type = request.form['fabric_type']
    color = request.form['color']
    category = request.form['category']
    buyer_name = request.form['buyer_name']
    qty = request.form.get('qty_moq', '')
    season = request.form.get('season', '')
    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(url_for('admin_panel'))
    image = request.files['image']
    if image.filename == '':
        flash('No image selected')
        return redirect(url_for('admin_panel'))
    image_filename = secure_filename(f"{product_id}_{image.filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image.save(image_path)
    image_url = f"/uploads/{image_filename}"
    new_row = {
        'DBID': df['DBID'].max() + 1 if not df.empty else 1,
        'Product ID': product_id,
        'Description': description,
        'Fabric Type': fabric_type,
        'Color': color,
        'Category': category,
        'Buyer Name': buyer_name,
        'QTY (MOQ)': qty,
        'Season': season,
        'Image_URL': image_url
    }
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    df_filtered = df.copy()
    df.to_excel('new_garments_with_urls.xlsx', index=False)
    image_urls.append(image_url)
    descriptions.append(description)
    fabric_types.append(fabric_type)
    colors.append(color)
    patterns.append('')
    buyer_names.append(buyer_name)
    qty_moq.append(qty)
    seasons.append(season)
    new_image_features = extract_image_features(image_path)
    new_text_features = extract_text_features(description, fabric_type, color, '', buyer_name, qty, season)
    if new_image_features is not None:
        image_features.append(new_image_features)
    text_features.append(new_text_features)
    with open(cache_file, 'wb') as f:
        pickle.dump((image_features, text_features), f)
    flash('Garment added successfully!')
    return redirect(url_for('admin_panel'))

@app.route('/download_template')
def download_template():
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    template_path = 'garment_upload_template.xlsx'
    if not os.path.exists(template_path):
        create_excel_template()
    return send_file(template_path, as_attachment=True, download_name='garment_upload_template.xlsx')

@app.route('/bulk_upload', methods=['GET', 'POST'])
def bulk_upload():
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    global df, df_filtered, image_urls, descriptions, fabric_types, colors, patterns, image_features, text_features, buyer_names, qty_moq, seasons
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('admin_panel'))
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.xlsx'):
            flash('Please upload a valid Excel file (.xlsx)')
            return redirect(url_for('admin_panel'))
        temp_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('temp_upload.xlsx'))
        file.save(temp_excel_path)
        try:
            new_data = pd.read_excel(temp_excel_path)
        except Exception as e:
            flash(f'Error reading Excel file: {str(e)}')
            os.remove(temp_excel_path)
            return redirect(url_for('admin_panel'))
        required_columns = ['Product ID', 'Description', 'Fabric Type', 'Color', 'Category', 'Buyer Name']
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            flash(f'Missing required columns: {", ".join(missing_columns)}')
            os.remove(temp_excel_path)
            return redirect(url_for('admin_panel'))
        if 'DBID' not in new_data.columns:
            new_data['DBID'] = pd.Series(dtype=int)
        new_data['DBID'] = new_data['DBID'].fillna(0).astype(int)
        max_dbid = df['DBID'].max() if not df.empty else 0
        for idx in new_data.index:
            if new_data.at[idx, 'DBID'] == 0:
                max_dbid += 1
                new_data.at[idx, 'DBID'] = max_dbid
        new_data = new_data.fillna('')
        wb = load_workbook(temp_excel_path)
        ws = wb.active
        image_paths = []
        for idx, row in new_data.iterrows():
            image_found = False
            for image in ws._images:
                cell = image.anchor._from
                if cell.row == idx + 2:
                    image_filename = secure_filename(f"bulk_{row['Product ID']}_{idx}.jpg")
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                    with open(image_path, 'wb') as f:
                        f.write(image._data())
                    image_paths.append(f"/uploads/{image_filename}")
                    image_found = True
                    break
            if not image_found:
                flash(f'No image found for row {idx + 2}.')
                os.remove(temp_excel_path)
                return redirect(url_for('admin_panel'))
        new_data['Image_URL'] = image_paths
        df = pd.concat([df, new_data], ignore_index=True)
        df_filtered = df.copy()
        df.to_excel('new_garments_with_urls.xlsx', index=False)
        image_urls.extend(new_data['Image_URL'].tolist())
        descriptions.extend(new_data['Description'].tolist())
        fabric_types.extend(new_data['Fabric Type'].tolist())
        colors.extend(new_data['Color'].tolist())
        patterns.extend([''] * len(new_data))
        buyer_names.extend(new_data['Buyer Name'].tolist())
        qty_moq.extend(new_data.get('QTY (MOQ)', pd.Series([''] * len(new_data))).tolist())
        seasons.extend(new_data.get('Season', pd.Series([''] * len(new_data))).tolist())
        new_image_features = [extract_image_features(os.path.join(app.config['UPLOAD_FOLDER'], path.split('/')[-1])) for path in image_paths]
        new_text_features = [extract_text_features(desc, fab, col, '', buy, qty, seas) 
                            for desc, fab, col, buy, qty, seas in zip(new_data['Description'], new_data['Fabric Type'], new_data['Color'], new_data['Buyer Name'], new_data.get('QTY (MOQ)', pd.Series([''])), new_data.get('Season', pd.Series([''])))]
        image_features.extend([f for f in new_image_features if f is not None])
        text_features.extend(new_text_features)
        with open(cache_file, 'wb') as f:
            pickle.dump((image_features, text_features), f)
        os.remove(temp_excel_path)
        flash(f'Successfully added {len(new_data)} garments!')
        return redirect(url_for('admin_panel'))
    return redirect(url_for('admin_panel'))

@app.route('/edit_garment/<int:dbid>', methods=['GET', 'POST'])
def edit_garment(dbid):
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    global df, df_filtered, image_urls, descriptions, fabric_types, colors, patterns, image_features, text_features, buyer_names, qty_moq, seasons
    garment = df[df['DBID'] == dbid]
    if garment.empty:
        flash('Garment not found')
        return redirect(url_for('admin_panel'))
    garment = garment.iloc[0]
    if request.method == 'POST':
        product_id = request.form['product_id']
        description = request.form['description']
        fabric_type = request.form['fabric_type']
        color = request.form['color']
        category = request.form['category']
        buyer_name = request.form['buyer_name']
        qty = request.form.get('qty_moq', garment.get('QTY (MOQ)', ''))
        season = request.form.get('season', garment.get('Season', ''))
        image_url = garment['Image_URL']
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image']
            image_filename = secure_filename(f"{product_id}_{image.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image.save(image_path)
            image_url = f"/uploads/{image_filename}"
            old_image_path = os.path.join(app.config['UPLOAD_FOLDER'], garment['Image_URL'].split('/')[-1])
            if os.path.exists(old_image_path):
                os.remove(old_image_path)
        df.loc[df['DBID'] == dbid, 'Product ID'] = product_id
        df.loc[df['DBID'] == dbid, 'Description'] = description
        df.loc[df['DBID'] == dbid, 'Fabric Type'] = fabric_type
        df.loc[df['DBID'] == dbid, 'Color'] = color
        df.loc[df['DBID'] == dbid, 'Category'] = category
        df.loc[df['DBID'] == dbid, 'Buyer Name'] = buyer_name
        df.loc[df['DBID'] == dbid, 'QTY (MOQ)'] = qty
        df.loc[df['DBID'] == dbid, 'Season'] = season
        df.loc[df['DBID'] == dbid, 'Image_URL'] = image_url
        df_filtered = df.copy()
        df.to_excel('new_garments_with_urls.xlsx', index=False)
        idx = df.index[df['DBID'] == dbid][0]
        image_urls[idx] = image_url
        descriptions[idx] = description
        fabric_types[idx] = fabric_type
        colors[idx] = color
        buyer_names[idx] = buyer_name
        qty_moq[idx] = qty
        seasons[idx] = season
        new_image_features = extract_image_features(os.path.join(app.config['UPLOAD_FOLDER'], image_url.split('/')[-1]))
        new_text_features = extract_text_features(description, fabric_type, color, '', buyer_name, qty, season)
        if new_image_features is not None:
            image_features[idx] = new_image_features
        text_features[idx] = new_text_features
        with open(cache_file, 'wb') as f:
            pickle.dump((image_features, text_features), f)
        flash('Garment updated successfully!')
        return redirect(url_for('admin_panel'))
    return render_template('edit_garment.html', garment=garment)

@app.route('/delete_garment/<int:dbid>')
def delete_garment(dbid):
    if 'user_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('login'))
    global df, df_filtered, image_urls, descriptions, fabric_types, colors, patterns, image_features, text_features, buyer_names, qty_moq, seasons
    idx = df.index[df['DBID'] == dbid]
    if idx.empty:
        flash('Garment not found')
        return redirect(url_for('admin_panel'))
    idx = idx[0]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], df.iloc[idx]['Image_URL'].split('/')[-1])
    if os.path.exists(image_path):
        os.remove(image_path)
    df = df.drop(idx).reset_index(drop=True)
    df_filtered = df.copy()
    image_urls.pop(idx)
    descriptions.pop(idx)
    fabric_types.pop(idx)
    colors.pop(idx)
    patterns.pop(idx)
    buyer_names.pop(idx)
    qty_moq.pop(idx)
    seasons.pop(idx)
    image_features.pop(idx)
    text_features.pop(idx)
    df.to_excel('new_garments_with_urls.xlsx', index=False)
    with open(cache_file, 'wb') as f:
        pickle.dump((image_features, text_features), f)
    flash('Garment deleted successfully!')
    return redirect(url_for('admin_panel'))

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    selected_items = session.get('selected_items', [])
    if not selected_items:
        flash('No items selected for the PDF report.')
        return redirect(url_for('results'))
    results = session.get('results', [])
    if not results:
        flash('No results available to generate a PDF.')
        return redirect(url_for('results'))
    selected_results = [(url, info) for url, info in results if str(info['DBID']) in selected_items]
    if not selected_results:
        flash('Selected items not found in results.')
        return redirect(url_for('results'))
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Garment Recognition System - Search Results")
    y_position -= 30
    for url, info in selected_results:
        try:
            response = http_session.get(url, timeout=10)
            if response.status_code != 200:
                continue
            img_data = BytesIO(response.content)
            img = ImageReader(img_data)
            img_width, img_height = 200, 200
            if y_position - img_height < 50:
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, y_position, "Garment Recognition System - Search Results")
                y_position -= 30
            c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
            x_text = 270
            c.setFont("Helvetica", 12)
            c.drawString(x_text, y_position - 20, f"Product ID: {info['Product ID']}")
            c.drawString(x_text, y_position - 40, f"Description: {info['Description']}")
            c.drawString(x_text, y_position - 60, f"Fabric: {info['Fabric Type']}")
            c.drawString(x_text, y_position - 80, f"Color: {info['Color']}")
            c.drawString(x_text, y_position - 100, f"Category: {info['Category']}")
            if 'Buyer Name' in info:
                c.drawString(x_text, y_position - 120, f"Buyer Name: {info['Buyer Name']}")
            if 'QTY (MOQ)' in info:
                c.drawString(x_text, y_position - 140, f"QTY (MOQ): {info['QTY (MOQ)']}")
            if 'Season' in info:
                c.drawString(x_text, y_position - 160, f"Season: {info['Season']}")
            y_position -= (img_height + 30)
        except Exception as e:
            print(f"Error processing image for PDF: {str(e)}")
            continue
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='garment_report.pdf', mimetype='application/pdf')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)