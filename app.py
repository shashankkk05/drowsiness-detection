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
import requests  # FIXED: Added at top level

app = Flask(__name__)
CORS(app)

# FIXED: Added thread locks for safety
stats_lock = Lock()
email_lock = Lock()

# Global variables
try:
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
    model = load_model("drowiness_new7.h5")
    print("✅ Model and cascades loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or cascades: {e}")
    print("Make sure you have:")
    print("  - drowiness_new7.h5 in the root folder")
    print("  - data/haarcascade_*.xml files")
    exit(1)

camera = None
detection_active = False
count = 0
alarm_on = False
stats = {
    'eyes_closed_count': 0,
    'total_frames': 0,
    'drowsiness_events': 0,
    'alarm_triggered': False
}

# Email configuration
email_config = {
    'recipient_email': '',
    'email_sent': False
}

def send_sos_email():
    """Send SOS email notification using Formspree"""
    global email_config, stats
    
    with email_lock:
        if not email_config['recipient_email'] or email_config['email_sent']:
            print("⚠️ Email not sent - either no recipient or already sent")
            return
        
        recipient = email_config['recipient_email']
    
    try:
        # Your Formspree form ID
        formspree_url = "https://formspree.io/f/mjkjzeza"
        
        with stats_lock:
            # Formspree expects these field names
            email_data = {
                "_replyto": recipient,  # This sets the reply-to address
                "email": recipient,      # Recipient email
                "_subject": "🚨 DROWSINESS ALERT - IMMEDIATE ATTENTION REQUIRED",
                "message": f"""
DROWSINESS ALERT - IMMEDIATE ATTENTION REQUIRED!

The drowsiness detection system has detected critical drowsiness levels.

EVENT DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⚠️ Drowsiness Events: {stats['drowsiness_events']}
👁️ Eyes Closed Count: {stats['eyes_closed_count']}
📊 Total Frames: {stats['total_frames']}

RECOMMENDED ACTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Contact the driver immediately
2. Ensure driver takes a break
3. Check driver's location and status
4. Do not ignore - drowsy driving is dangerous!

This is an automated alert from the Drowsiness Detection System.
                """,
                "drowsiness_events": str(stats['drowsiness_events']),
                "eyes_closed_count": str(stats['eyes_closed_count']),
                "total_frames": str(stats['total_frames'])
            }
        
        print("\n" + "="*60)
        print("🚨 SOS EMAIL ALERT TRIGGERED!")
        print("="*60)
        print(f"📧 Sending to: {recipient}")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Send email via Formspree with proper headers
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            formspree_url, 
            json=email_data,  # Use json instead of data
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ EMAIL SENT SUCCESSFULLY via Formspree!")
            with email_lock:
                email_config['email_sent'] = True
        elif response.status_code == 422:
            print(f"⚠️ Formspree validation error: {response.text}")
            print("💡 Check that your Formspree form accepts these fields")
        else:
            print(f"⚠️ Formspree response code: {response.status_code}")
            print(f"Response: {response.text}")
        
        print("="*60 + "\n")
        
    except requests.exceptions.Timeout:
        print(f"\n❌ Email sending timeout - Formspree took too long to respond")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Email sending failed: {str(e)}")
        print("\n💡 TROUBLESHOOTING:")
        print("="*60)
        print("   1. Make sure you're connected to the internet")
        print("   2. Verify your Formspree form ID is correct: mjkjzeza")
        print("   3. Check Formspree dashboard for any errors")
        print("   4. Install requests: pip install requests")
        print("="*60 + "\n")

def start_alarm():
    """Set alarm flag and check for SOS email"""
    global stats, alarm_on
    
    if not alarm_on:
        with stats_lock:
            stats['alarm_triggered'] = True
            stats['drowsiness_events'] += 1  # FIXED: Moved here
            current_events = stats['drowsiness_events']
        
        alarm_on = True
        print(f"⚠️ DROWSINESS ALERT! Event #{current_events}")
        
        with email_lock:
            email_sent = email_config['email_sent']
        
        # Send email after 3 events
        if current_events >= 3 and not email_sent:
            print(f"🚨 SOS THRESHOLD REACHED (3 events)! Sending notification...")
            Thread(target=send_sos_email, daemon=True).start()

def generate_frames():
    global camera, detection_active, count, alarm_on, stats
    
    camera = cv2.VideoCapture(0)
    
    # FIXED: Check if camera opened successfully
    if not camera.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    status1 = 0
    status2 = 0
    
    while detection_active:
        success, frame = camera.read()
        if not success:
            print("⚠️ Failed to read frame")
            break
        
        with stats_lock:
            stats['total_frames'] += 1
        
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
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

            # Check if both eyes are closed (status 2 = closed)
            if status1 == 2 and status2 == 2:
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
                # Eyes are open
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
    
    # FIXED: Proper camera cleanup
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
    global detection_active, count, alarm_on
    
    detection_active = True
    count = 0
    alarm_on = False
    
    with stats_lock:
        stats['eyes_closed_count'] = 0
        stats['total_frames'] = 0
        stats['drowsiness_events'] = 0
        stats['alarm_triggered'] = False
    
    with email_lock:
        email_config['email_sent'] = False
    
    print("\n✅ Detection started!")
    print(f"📧 SOS Email: {email_config.get('recipient_email', 'Not set')}")
    print(f"🚨 Email will be sent after 3 drowsiness events\n")
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global detection_active, camera
    detection_active = False
    time.sleep(1)  # FIXED: Increased delay for proper cleanup
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
    """Serve the alarm file from static folder"""
    return send_from_directory('static', 'alarm.mp3')

@app.route('/sos/email', methods=['POST'])
def set_sos_email():
    """Set SOS email address"""
    data = request.json
    email = data.get('email', '')
    
    with email_lock:
        email_config['recipient_email'] = email
    
    print(f"\n✅ SOS Email configured: {email}")
    print(f"🚨 Email will be sent after 3 drowsiness events\n")
    return jsonify({'status': 'success', 'email': email})

@app.route('/sos/email', methods=['GET'])
def get_sos_email():
    """Get SOS email address"""
    with email_lock:
        return jsonify({'email': email_config['recipient_email']})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 DROWSINESS DETECTION SYSTEM WITH SOS EMAIL")
    print("="*60)
    print("\n✅ Server starting...")
    print("🌐 Open your browser: http://127.0.0.1:5000")
    print("\n📧 Features:")
    print("  • Real-time drowsiness detection")
    print("  • Alarm sounds when eyes closed for 5+ frames")
    print("  • 🚨 SOS email notification after 3 drowsiness events")
    print("\n⚠️  Press CTRL+C to stop\n")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, use_reloader=False)
