# TELEGRAM BOT SOLUTION - WITH NAME/PHONE & DEBUGGING

from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from threading import Thread, Lock
import time
import os
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

# ==================== TELEGRAM CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = "8548717830:AAFxmnkIexaCFntNAhQA_vg-IifnQBwhQEs"
TELEGRAM_CHAT_ID = "6555326135"
# ================================================================

# --- Global variables ---
stats_lock = Lock()
telegram_lock = Lock()
user_info_lock = Lock()

try:
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
    model = load_model("drowiness_new7.h5")
    print("✅ Model and cascades loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or cascades: {e}")
    exit(1)

camera = None
detection_active = False
count = 0
alarm_on = False

# Stats and user info dictionaries
stats = {
    'eyes_closed_count': 0,
    'total_frames': 0,
    'drowsiness_events': 0,
    'alarm_triggered': False
}
telegram_config = {
    'alert_sent': False
}
user_info = {
    'name': 'N/A',
    'phone': 'N/A'
}
# -------------------------

def send_telegram_alert():
    """Send alert via Telegram"""
    global telegram_config, stats, user_info
    
    with telegram_lock:
        if telegram_config['alert_sent']:
            print("⚠️ Alert already sent")
            return
    
    try:
        # Get user info safely
        with user_info_lock:
            name = user_info['name']
            phone = user_info['phone']
            
        with stats_lock:
            message = f"""
🚨 *DROWSINESS ALERT*
⚠️ IMMEDIATE ATTENTION REQUIRED!

👤 *Driver Info:*
  • *Name:* {name}
  • *Phone:* {phone}

📊 *Event Details:*
  • *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  • *Drowsiness Events:* {stats['drowsiness_events']}
  • *Eyes Closed Count:* {stats['eyes_closed_count']}

🚨 *RECOMMENDED ACTIONS:*
1️⃣ Contact the driver immediately
2️⃣ Ensure driver takes a break
"""
        
        print("\n" + "="*60)
        print("🚨 TELEGRAM ALERT TRIGGERED!")
        print("="*60)
        print(f"📱 Sending to Chat ID: {TELEGRAM_CHAT_ID}")
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ TELEGRAM ALERT SENT SUCCESSFULLY!")
            with telegram_lock:
                telegram_config['alert_sent'] = True
        else:
            print(f"⚠️ Telegram API response: {response.status_code}")
            print(f"Error: {response.text}")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Telegram alert failed: {str(e)}")


def start_alarm():
    """Set alarm flag and check for alert"""
    global stats, alarm_on
    
    if not alarm_on:
        with stats_lock:
            stats['alarm_triggered'] = True
            stats['drowsiness_events'] += 1
            current_events = stats['drowsiness_events']
        
        alarm_on = True
        print(f"⚠️ DROWSINESS ALERT! Event #{current_events}")
        
        with telegram_lock:
            alert_sent = telegram_config['alert_sent']
        
        if current_events >= 3 and not alert_sent:
            print(f"🚨 ALERT THRESHOLD REACHED (3 events)! Sending notification...")
            Thread(target=send_telegram_alert, daemon=True).start()

def generate_frames():
    global camera, detection_active, count, alarm_on, stats
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            break
        
        with stats_lock:
            stats['total_frames'] += 1
        
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR_GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # ==========================================================
            #  LOGIC FIX: Default to 'Open' (1) each frame
            #  This prevents using old data if eyes aren't found
            # ==========================================================
            status1 = 1  # 1 = Open
            status2 = 1  # 1 = Open
            
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)
            
            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                if eye1.size > 0:
                    eye1 = cv2.resize(eye1, (145, 145))
                    eye1 = eye1.astype('float') / 255.0
                    eye1 = img_to_array(eye1)
                    eye1 = np.expand_dims(eye1, axis=0)
                    pred1 = model.predict(eye1, verbose=0)
                    status1 = np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                if eye2.size > 0:
                    eye2 = cv2.resize(eye2, (145, 145))
                    eye2 = eye2.astype('float') / 255.0
                    eye2 = img_to_array(eye2)
                    eye2 = np.expand_dims(eye2, axis=0)
                    pred2 = model.predict(eye2, verbose=0)
                    status2 = np.argmax(pred2)
                break
            
            # ==========================================================
            #  CRITICAL DEBUG LINE:
            #  This will print the status of both eyes to your terminal.
            # ==========================================================
            print(f"DEBUG: Left Eye={status1}, Right Eye={status2}")

            # This is our current 'guess' (0 = Closed). 
            # The DEBUG print will tell us if this is correct.
            if status1 == 0 and status2 == 0:
                count += 1
                with stats_lock:
                    stats['eyes_closed_count'] += 1
                
                cv2.putText(frame, f"Eyes Closed: {count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if count >= 5:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not alarm_on:
                        t = Thread(target=start_alarm)
                        t.daemon = True
                        t.start()
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                count = 0
                alarm_on = False
                with stats_lock:
                    stats['alarm_triggered'] = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if camera and camera.isOpened():
        camera.release()
        print("📹 Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection():
    global detection_active, count, alarm_on, user_info
    
    # Get user info from the POST request
    data = request.json
    with user_info_lock:
        user_info['name'] = data.get('name', 'N/A')
        user_info['phone'] = data.get('phone', 'N/A')
    
    # Reset stats
    detection_active = True
    count = 0
    alarm_on = False
    
    with stats_lock:
        stats['eyes_closed_count'] = 0
        stats['total_frames'] = 0
        stats['drowsiness_events'] = 0
        stats['alarm_triggered'] = False
    
    with telegram_lock:
        telegram_config['alert_sent'] = False
    
    print("\n✅ Detection started!")
    print(f"👤 User: {user_info['name']}, {user_info['phone']}")
    print(f"🚨 Alert will be sent after 3 drowsiness events\n")
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global detection_active, camera
    detection_active = False
    time.sleep(1)
    if camera and camera.isOpened():
        camera.release()
    print("\n🛑 Detection stopped\n")
    return jsonify({'status': 'stopped'})

@app.route('/stats')
def get_stats():
    with stats_lock:
        return jsonify(dict(stats))

@app.route('/static/alarm.mp3')
def serve_alarm():
    return send_from_directory('static', 'alarm.mp3')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 DROWSINESS DETECTION - TELEGRAM ALERTS")
    print("="*60)
    print("\n✅ Server starting...")
    print("🌐 Open your browser: http://127.0.0.1:5000")
    print("\n⚠️  Press CTRL+C to stop\n")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, use_reloader=False)