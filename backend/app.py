from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
import random, time
import numpy as np
from fraud_model import predict

app = Flask(__name__)
CORS(app)

# --------------------- CONFIG ---------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'wattoohuzaifa18@gmail.com'  # Your email
app.config['MAIL_PASSWORD'] = 'gnfoneyfsajohenk'  # App password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

otp_storage = {}
user_passwords = {}  # Simulated in-memory password store

# ----------------- HELPERS -----------------
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

# ----------------- FRAUD PREDICTION -----------------
@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    try:
        result = predict(data)
        result = convert_numpy_types(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ----------------- SEND OTP -----------------
@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400

    otp = str(random.randint(100000, 999999))
    otp_storage[email] = {'otp': otp, 'timestamp': time.time()}

    try:
        msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f'Your OTP code is: {otp}. It will expire in 5 minutes.'
        mail.send(msg)
        return jsonify({'success': True, 'message': 'OTP sent successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to send OTP: {str(e)}'}), 500

# ----------------- RESET PASSWORD -----------------
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')
    new_password = data.get('newPassword')

    if not all([email, otp, new_password]):
        return jsonify({'success': False, 'message': 'All fields are required'}), 400

    entry = otp_storage.get(email)

    if not entry:
        return jsonify({'success': False, 'message': 'No OTP found'}), 400

    if time.time() - entry['timestamp'] > 300:
        del otp_storage[email]
        return jsonify({'success': False, 'message': 'OTP expired'}), 400

    if entry['otp'] != otp:
        return jsonify({'success': False, 'message': 'Invalid OTP'}), 400

    user_passwords[email] = new_password  # NOTE: In real apps, hash this
    del otp_storage[email]

    return jsonify({'success': True, 'message': 'Password reset successful'})

# ----------------- MAIN -----------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


