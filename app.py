from datetime import datetime
from functools import wraps
import json
import sqlite3
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.secret_key = "crop-yield-secret-key-change-this"

MODEL_FILE = Path("model.pkl")
DATASET_FILE = Path("dataset.csv")
DB_FILE = Path("app_data.db")
OFFICIAL_LOCATIONS_CACHE = Path("india_state_districts.json")
ADMIN_INVITE_CODE = "CROP-ADMIN-2026"

STATE_DISTRICT_MAP = {
    "Andhra Pradesh": ["Visakhapatnam", "Guntur", "Kurnool", "Nellore"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Pasighat", "Ziro"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Korba"],
    "Goa": ["North Goa", "South Goa", "Panaji", "Margao"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Haryana": ["Karnal", "Hisar", "Gurugram", "Ambala"],
    "Himachal Pradesh": ["Shimla", "Kullu", "Kangra", "Mandi"],
    "Jharkhand": ["Ranchi", "Dhanbad", "Bokaro", "Jamshedpur"],
    "Karnataka": ["Mysuru", "Bengaluru Urban", "Belagavi", "Dharwad"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur"],
    "Maharashtra": ["Nashik", "Nagpur", "Pune", "Aurangabad"],
    "Manipur": ["Imphal West", "Imphal East", "Thoubal", "Churachandpur"],
    "Meghalaya": ["East Khasi Hills", "West Garo Hills", "Ri Bhoi", "Jaintia Hills"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Serchhip"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Wokha"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Puri", "Sambalpur"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Jalandhar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Sikkim": ["Gangtok", "Namchi", "Gyalshing", "Mangan"],
    "Tamil Nadu": ["Coimbatore", "Chennai", "Madurai", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
    "Tripura": ["Agartala", "Dhalai", "Unakoti", "Sepahijala"],
    "Uttar Pradesh": ["Kanpur", "Lucknow", "Varanasi", "Agra"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Nainital", "Udham Singh Nagar"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Bardhaman"],
    "Andaman and Nicobar Islands": ["South Andaman", "North and Middle Andaman", "Nicobar"],
    "Chandigarh": ["Chandigarh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Daman", "Diu", "Dadra and Nagar Haveli"],
    "Delhi": ["New Delhi", "Central Delhi", "North Delhi", "South Delhi"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla"],
    "Ladakh": ["Leh", "Kargil"],
    "Lakshadweep": ["Kavaratti", "Agatti", "Minicoy"],
    "Puducherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"],
}

model_bundle = None
model = None
model_name = "Not loaded"
feature_columns = []


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                crop_year INTEGER NOT NULL,
                season TEXT NOT NULL,
                crop TEXT NOT NULL,
                area REAL NOT NULL,
                production REAL NOT NULL,
                rainfall REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                soil_type TEXT NOT NULL,
                fertilizer_usage REAL NOT NULL,
                predicted_yield REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        admin_email = "admin@cropyield.local"
        admin_password = generate_password_hash("Admin@12345")
        existing_admin = connection.execute(
            "SELECT id FROM users WHERE email = ?", (admin_email,)
        ).fetchone()
        if existing_admin is None:
            connection.execute(
                """
                INSERT INTO users (name, email, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "Admin",
                    admin_email,
                    admin_password,
                    "admin",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        connection.commit()


def load_model_bundle() -> None:
    global model_bundle, model, model_name, feature_columns

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Run train_model.py first to train and save the model."
        )

    model_bundle = joblib.load(MODEL_FILE)
    model = model_bundle["model"]
    model_name = model_bundle.get("model_name", "Best Model")
    feature_columns = model_bundle.get("feature_columns", [])


def fetch_state_district_map() -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }

    # Try the public CoWIN API first.
    try:
        states_request = Request(
            "https://cdn-api.co-vin.in/api/v2/admin/location/states",
            headers=headers,
        )
        with urlopen(states_request, timeout=20) as response:
            states_payload = json.loads(response.read().decode("utf-8"))

        state_district_map = {}
        for state in states_payload.get("states", []):
            state_id = state.get("state_id")
            state_name = str(state.get("state_name", "")).strip()
            if not state_id or not state_name:
                continue

            district_request = Request(
                f"https://cdn-api.co-vin.in/api/v2/admin/location/districts/{state_id}",
                headers=headers,
            )
            with urlopen(district_request, timeout=20) as response:
                district_payload = json.loads(response.read().decode("utf-8"))

            district_names = sorted(
                {
                    str(district.get("district_name", "")).strip()
                    for district in district_payload.get("districts", [])
                    if str(district.get("district_name", "")).strip()
                }
            )

            if district_names:
                state_district_map[state_name] = district_names

        if state_district_map:
            return state_district_map
    except Exception:
        pass

    # Public fallback dataset if the government API is blocked.
    fallback_request = Request(
        "https://raw.githubusercontent.com/sab99r/Indian-States-And-Districts/master/states-and-districts.json",
        headers=headers,
    )
    with urlopen(fallback_request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    state_district_map = {}
    for state_entry in payload.get("states", []):
        state_name = str(state_entry.get("state", "")).strip()
        district_names = sorted(
            {
                str(district).strip()
                for district in state_entry.get("districts", [])
                if str(district).strip()
            }
        )
        if state_name and district_names:
            state_district_map[state_name] = district_names

    return state_district_map


def load_state_district_map() -> dict:
    if OFFICIAL_LOCATIONS_CACHE.exists():
        try:
            cached_map = json.loads(OFFICIAL_LOCATIONS_CACHE.read_text(encoding="utf-8"))
            if isinstance(cached_map, dict) and cached_map:
                return {k: sorted(set(v)) for k, v in cached_map.items() if isinstance(v, list)}
        except Exception:
            pass

    try:
        state_district_map = fetch_state_district_map()
        if state_district_map:
            OFFICIAL_LOCATIONS_CACHE.write_text(
                json.dumps(state_district_map, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            return state_district_map
    except Exception:
        pass

    return STATE_DISTRICT_MAP


def get_dropdown_options() -> dict:
    state_district_map = load_state_district_map()
    all_states = sorted(state_district_map.keys())
    all_districts = sorted({district for districts in state_district_map.values() for district in districts})

    fallback = {
        "states": all_states,
        "districts": all_districts,
        "seasons": ["Kharif", "Rabi", "Zaid"],
        "crops": ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean"],
        "soil_types": ["Alluvial", "Black", "Red", "Laterite", "Clay", "Sandy"],
        "state_district_map": state_district_map,
    }

    if not DATASET_FILE.exists():
        return fallback

    try:
        df = pd.read_csv(DATASET_FILE)
        dataset_states = sorted(df["State"].dropna().astype(str).unique().tolist())
        merged_states = sorted(set(all_states + dataset_states))
        merged_map = {state: list(districts) for state, districts in state_district_map.items()}

        state_district_df = df[["State", "District"]].dropna().copy()
        state_district_df["State"] = state_district_df["State"].astype(str)
        state_district_df["District"] = state_district_df["District"].astype(str)
        for state_name, group in state_district_df.groupby("State"):
            dataset_districts = sorted(group["District"].unique().tolist())
            merged_map[state_name] = sorted(set(merged_map.get(state_name, []) + dataset_districts))

        return {
            "states": merged_states or fallback["states"],
            "districts": sorted(df["District"].dropna().astype(str).unique().tolist())
            or fallback["districts"],
            "seasons": sorted(df["Season"].dropna().astype(str).unique().tolist()) or fallback["seasons"],
            "crops": sorted(df["Crop"].dropna().astype(str).unique().tolist()) or fallback["crops"],
            "soil_types": sorted(df["Soil_Type"].dropna().astype(str).unique().tolist())
            or fallback["soil_types"],
            "state_district_map": merged_map,
        }
    except Exception:
        return fallback


def parse_positive_number(value: str, field_name: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc

    if number < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return number


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def admin_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access is required for that page.", "error")
            return redirect(url_for("index"))
        return view_function(*args, **kwargs)

    return wrapped_view


def get_current_user() -> sqlite3.Row | None:
    user_id = session.get("user_id")
    if user_id is None:
        return None
    with get_db_connection() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


@app.context_processor
def inject_user_context() -> dict:
    return {
        "current_user": get_current_user(),
        "current_role": session.get("role"),
    }


@app.route("/")
@login_required
def index():
    options = get_dropdown_options()
    selected_state = request.args.get("state", "").strip() or options["states"][0]
    districts_for_state = options["state_district_map"].get(selected_state, options["districts"])
    selected_district = request.args.get("district", "").strip() or (districts_for_state[0] if districts_for_state else "")

    with get_db_connection() as connection:
        history_rows = connection.execute(
            """
            SELECT predicted_yield, state, district, crop, season, area, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (session["user_id"],),
        ).fetchall()

    return render_template(
        "index.html",
        prediction=None,
        error_message=None,
        model_name=model_name,
        options=options,
        selected_state=selected_state,
        selected_district=selected_district,
        history=history_rows,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    options = get_dropdown_options()
    selected_state = request.form.get("state", "").strip()
    selected_district = request.form.get("district", "").strip()
    districts_for_state = options["state_district_map"].get(selected_state, options["districts"])
    if not selected_district:
        selected_district = districts_for_state[0] if districts_for_state else ""

    error_message = None
    prediction = None

    try:
        if model is None:
            raise RuntimeError("Model is not loaded. Run train_model.py first.")

        form_data = {
            "State": selected_state,
            "District": selected_district,
            "Crop_Year": int(parse_positive_number(request.form.get("crop_year", ""), "Crop Year")),
            "Season": request.form.get("season", "").strip(),
            "Crop": request.form.get("crop", "").strip(),
            "Area": parse_positive_number(request.form.get("area", ""), "Area"),
            "Production": parse_positive_number(request.form.get("production", ""), "Production"),
            "Rainfall": parse_positive_number(request.form.get("rainfall", ""), "Rainfall"),
            "Temperature": parse_positive_number(request.form.get("temperature", ""), "Temperature"),
            "Humidity": parse_positive_number(request.form.get("humidity", ""), "Humidity"),
            "Soil_Type": request.form.get("soil_type", "").strip(),
            "Fertilizer_Usage": parse_positive_number(request.form.get("fertilizer_usage", ""), "Fertilizer Usage"),
        }

        input_df = pd.DataFrame([form_data])
        if feature_columns:
            input_df = input_df.reindex(columns=feature_columns)

        predicted_yield = model.predict(input_df)[0]
        prediction = round(float(predicted_yield), 3)

        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    user_id, state, district, crop_year, season, crop, area,
                    production, rainfall, temperature, humidity, soil_type,
                    fertilizer_usage, predicted_yield, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session["user_id"],
                    form_data["State"],
                    form_data["District"],
                    form_data["Crop_Year"],
                    form_data["Season"],
                    form_data["Crop"],
                    form_data["Area"],
                    form_data["Production"],
                    form_data["Rainfall"],
                    form_data["Temperature"],
                    form_data["Humidity"],
                    form_data["Soil_Type"],
                    form_data["Fertilizer_Usage"],
                    prediction,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            connection.commit()

    except Exception as exc:
        error_message = f"Prediction failed: {exc}"

    with get_db_connection() as connection:
        history_rows = connection.execute(
            """
            SELECT predicted_yield, state, district, crop, season, area, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (session["user_id"],),
        ).fetchall()

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        model_name=model_name,
        options=options,
        selected_state=selected_state,
        selected_district=selected_district,
        history=history_rows,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id") is not None:
        return redirect(url_for("index"))

    options = get_dropdown_options()
    error_message = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "user")
        invite_code = request.form.get("invite_code", "").strip()

        if not name or not email or not password:
            error_message = "Name, email, and password are required."
        elif role not in {"user", "admin"}:
            error_message = "Invalid account type selected."
        elif role == "admin" and invite_code != ADMIN_INVITE_CODE:
            error_message = "Invalid admin invite code."
        else:
            try:
                with get_db_connection() as connection:
                    existing_user = connection.execute(
                        "SELECT id FROM users WHERE email = ?", (email,)
                    ).fetchone()
                    if existing_user is not None:
                        error_message = "An account with this email already exists."
                    else:
                        connection.execute(
                            """
                            INSERT INTO users (name, email, password_hash, role, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                name,
                                email,
                                generate_password_hash(password),
                                role,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            ),
                        )
                        connection.commit()
                        flash("Registration successful. Please log in.", "success")
                        return redirect(url_for("login"))
            except Exception as exc:
                error_message = f"Registration failed: {exc}"

    return render_template("register.html", error_message=error_message, options=options)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id") is not None:
        return redirect(url_for("index"))

    error_message = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            error_message = "Invalid email or password."
        else:
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("index"))

    return render_template("login.html", error_message=error_message)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db_connection() as connection:
        users = connection.execute(
            "SELECT id, name, email, role, created_at FROM users ORDER BY id DESC"
        ).fetchall()
        predictions = connection.execute(
            """
            SELECT p.created_at, u.name AS user_name, u.email, p.state, p.district, p.crop,
                   p.season, p.area, p.predicted_yield
            FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.id DESC
            LIMIT 50
            """
        ).fetchall()

    return render_template("admin.html", users=users, predictions=predictions)


@app.route("/profile")
@login_required
def profile():
    user = get_current_user()
    with get_db_connection() as connection:
        predictions = connection.execute(
            """
            SELECT predicted_yield, state, district, crop, season, area, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (session["user_id"],),
        ).fetchall()

    return render_template("profile.html", user=user, predictions=predictions)


@app.before_request
def seed_session_defaults() -> None:
    if request.endpoint in {"login", "register", "static"}:
        return
    if request.endpoint is None:
        return


if __name__ == "__main__":
    init_db()
    try:
        load_model_bundle()
    except Exception as startup_error:
        print(f"Startup warning: {startup_error}")

    app.run(host="0.0.0.0", port=5000, debug=True)