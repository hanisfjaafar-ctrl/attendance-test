from flask import Flask, render_template, jsonify, redirect, url_for, request
import subprocess
from threading import Thread
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import uuid
import base64
import io
import json
from PIL import Image
import numpy as np
import face_recognition
from datetime import datetime, time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

app = Flask(__name__)
CORS(app)

# ---------- Enhanced Geocoder for Place Names ----------
geolocator = Nominatim(user_agent="attendance_system_v2")

def get_place_name(lat, lng, max_retries=3):
    """
    Returns a human-friendly place name (e.g., 'Setapak Central Mall')
    Uses OpenStreetMap POI tags for best result.
    """
    for _ in range(max_retries):
        try:
            location = geolocator.reverse(
                f"{lat}, {lng}",
                language='en',
                zoom=18,
                addressdetails=True
            )
            if not location:
                return "Unknown location"

            addr = location.raw.get('address', {})
            
            place_name = (
                addr.get('building') or
                addr.get('amenity') or
                addr.get('shop') or
                addr.get('leisure') or
                addr.get('tourism') or
                addr.get('public_building') or
                addr.get('university') or
                addr.get('school') or
                (addr.get('house_number', '') + ' ' + addr.get('road', '')).strip() or
                location.address
            ).strip()

            return place_name if place_name else "Location not identified"
        except GeocoderTimedOut:
            continue
    return "Geocoding failed"

# ---------- Firebase ----------
cred_path = "serviceAccountKey.json"
if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Firebase credential file '{cred_path}' not found!")

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(credentials.Certificate(cred_path))

db = firestore.client()

scan_results = {}

# ---------- Configuration ----------
TOLERANCE = 0.45
ON_TIME_END = time(12, 30)  # On time if before or at 12:30 PM

# ---------- Frontend Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/admin-dashboard")
def admin_dashboard():
    return render_template("admin-dashboard.html")

@app.route("/admin-attendancelog")
def admin_attendancelog():
    return render_template("admin-attendancelog.html")

@app.route("/leavecalendar")
def leave_calendar():
    return render_template("leavecalendar.html")

@app.route("/admin-leavecalendar")
def admin_leave_calendar():
    return render_template("admin-leavecalendar.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/manage-staff")
def manage_staff():
    return render_template("manage-staff.html")

@app.route("/admin-liveloc")
def admin_liveloc():
    return render_template("admin-liveloc.html")

# ---------- API Routes ----------

@app.route("/get-attendance")
def get_attendance():
    try:
        docs = (
            db.collection("attendance_test")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
        records = []
        for doc in docs:
            data = doc.to_dict()
            records.append({
                "name": data.get("name", "Unknown"),
                "timestamp": data.get("timestamp", ""),
                "status": data.get("status", "Unknown"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "address": data.get("address", "N/A")
            })
        return jsonify(records)
    except Exception as e:
        print("❌ Error fetching attendance:", e)
        return jsonify([])

def run_face_recognition(scan_id):
    try:
        subprocess.run(["python", "live_recognition.py"])
        scan_results[scan_id] = {"status": "completed"}
    except Exception as e:
        scan_results[scan_id] = {"status": "failed", "error": str(e)}

@app.route("/scan")
def scan():
    scan_id = str(uuid.uuid4())
    Thread(target=run_face_recognition, args=(scan_id,)).start()
    return redirect(url_for("attendance", scan_id=scan_id))

@app.route("/scan-status/<scan_id>")
def scan_status(scan_id):
    return jsonify(scan_results.get(scan_id, {"status": "running"}))

@app.route("/clear-scan/<scan_id>")
def clear_scan(scan_id):
    scan_results.pop(scan_id, None)
    return jsonify({"status": "cleared"})

# ---------- Secure Recognition with Place Name ----------
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        user_uid = data.get("user_uid")
        if not user_uid:
            return jsonify({"success": False, "error": "User ID not provided"}), 400

        expected_name = data.get("expected_name", "").strip()
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if not expected_name:
            return jsonify({"success": False, "error": "User identity not provided"}), 400

        # Decode image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_image = np.array(image)

        # Load known faces
        if not os.path.exists("known_faces.json"):
            return jsonify({"success": False, "error": "No known faces enrolled"}), 400

        with open("known_faces.json", "r") as f:
            known_data = json.load(f)

        if not known_data.get("encodings") or not known_data.get("names"):
            return jsonify({"success": False, "error": "Known faces database is empty"}), 400

        known_encodings = [np.array(enc) for enc in known_data["encodings"]]
        known_names = known_data["names"]

        # Encode face
        unknown_encodings = face_recognition.face_encodings(rgb_image)
        if not unknown_encodings:
            return jsonify({"success": False, "error": "No face detected"}), 400

        unknown_encoding = unknown_encodings[0]
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_match_index = int(np.argmin(face_distances))
        recognized_name = known_names[best_match_index]
        best_distance = float(face_distances[best_match_index])

        # 🔒 Verify identity
        if recognized_name != expected_name:
            return jsonify({
                "success": False,
                "error": "Face does not match your account"
            }), 403

        if best_distance > TOLERANCE:
            return jsonify({"success": False, "error": "Face not recognized"}), 400

        # 🕒 Determine status
        now = datetime.now()
        current_time = now.time()
        status = "On Time" if current_time <= ON_TIME_END else "Late"

        # 🌍 Get PLACE NAME
        place_name = "Location not provided"
        if latitude is not None and longitude is not None:
            place_name = get_place_name(latitude, longitude)

        # 📥 Save to Firestore
        db.collection("attendance_test").add({
            "userId": user_uid,
            "name": recognized_name,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "latitude": latitude,
            "longitude": longitude,
            "address": place_name
        })

        return jsonify({
            "success": True,
            "name": recognized_name,
            "status": status,
            "address": place_name
        })
    except Exception as e:
        print("❌ Error in /recognize:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# ---------- NEW: Staff Live Location API ----------
@app.route("/api/staff-live-locations")
def get_staff_live_locations():
    """Fetch current live locations of staff from 'staff_locations' collection."""
    try:
        docs = db.collection("staff_locations").stream()
        locations = []
        for doc in docs:
            data = doc.to_dict()
            locations.append({
                "userId": doc.id,
                "name": data.get("name", "Unknown"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "lastUpdated": data.get("lastUpdated"),
                "status": data.get("status", "Offline")
            })
        return jsonify(locations)
    except Exception as e:
        print("❌ Error fetching live locations:", e)
        return jsonify([]), 500

# ---------- Optional: For testing location updates ----------
@app.route("/api/update-location", methods=["POST"])
def update_location():
    """Temporary endpoint to simulate staff location updates (for testing)."""
    try:
        data = request.get_json()
        user_id = data.get("userId")
        name = data.get("name")
        lat = data.get("latitude")
        lng = data.get("longitude")

        if not all([user_id, name, lat, lng]):
            return jsonify({"error": "Missing required fields"}), 400

        db.collection("staff_locations").document(user_id).set({
            "name": name,
            "latitude": lat,
            "longitude": lng,
            "lastUpdated": datetime.utcnow().isoformat(),
            "status": "Active"
        }, merge=True)

        return jsonify({"success": True})
    except Exception as e:
        print("❌ Error updating location:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)