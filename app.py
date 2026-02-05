from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import os

# ======================================================
# CONFIG
# ======================================================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MODEL_FILE = "learning_path_classifier.pkl"
ML_OVERRIDE_THRESHOLD = 0.65
MIN_AVG = 40

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ======================================================
# LOAD MODEL (plain or dict)
# ======================================================
model_data = joblib.load(MODEL_FILE)

if isinstance(model_data, dict):
    model = model_data.get("model", None)
    features_list = model_data.get("features", None)
    label_encoder = model_data.get("label_encoder", None)
else:
    model = model_data
    features_list = None
    label_encoder = None

# Debug: Print model features
print("=" * 50)
print("DEBUG: Model Information")
print("=" * 50)
if hasattr(model, 'feature_names_in_'):
    print(f"Model expects {len(model.feature_names_in_)} features:")
    for i, feat in enumerate(model.feature_names_in_, 1):
        print(f"  {i:3d}. {feat}")
    features_list = list(model.feature_names_in_)
elif features_list:
    print(f"Features list from saved model has {len(features_list)} features:")
    for i, feat in enumerate(features_list, 1):
        print(f"  {i:3d}. {feat}")
else:
    print("No feature names found in model")

print("=" * 50)

# ======================================================
# SUBJECTS (for manual input display)
# ======================================================
SUBJECTS = {
    "Mathematics": ["Mathematics_y1","Mathematics_y2","Mathematics_y3"],
    "Physics": ["Physics_y1","Physics_y2","Physics_y3"],
    "Chemistry": ["Chemistry_y1","Chemistry_y2","Chemistry_y3"],
    "Biology": ["Biology_Health_Sciences_y1","Biology_Health_Sciences_y2","Biology_Health_Sciences_y3"],
    "Geography": ["Geography_Environment_y1","Geography_Environment_y2","Geography_Environment_y3"],
    "Entrepreneurship": ["Entrepreneurship_y1","Entrepreneurship_y2","Entrepreneurship_y3"],
    "ICT": ["Information_and_Communication_Technology_y1","Information_and_Communication_Technology_y2","Information_and_Communication_Technology_y3"],
    "Economics": ["Entrepreneurship_y1","Entrepreneurship_y2","Entrepreneurship_y3"]
}

# ======================================================
# ALL SUBJECT COLUMNS - will be populated from model
# ======================================================
ALL_SUBJECT_COLUMNS = []

# If we have features_list from model, use it
if features_list:
    ALL_SUBJECT_COLUMNS = features_list
    print(f"\nUsing {len(ALL_SUBJECT_COLUMNS)} features from model")
else:
    # Fallback: Try to infer from error messages or common patterns
    print("\nWARNING: No features list found. This may cause errors.")

# ======================================================
# RULES AND WEIGHTS
# ======================================================
RULES = {
    "PCM":["Physics","Chemistry","Mathematics"],
    "PCB":["Physics","Chemistry","Biology"],
    "MPC":["Mathematics","Physics","ICT"],
    "MPG":["Mathematics","Physics","Geography"],
    "MCB":["Mathematics","Chemistry","Biology"],
    "BCG":["Biology","Chemistry","Geography"],
    "MEB":["Mathematics","Entrepreneurship","Economics"],
    "MEG":["Mathematics","Entrepreneurship","Geography"],
    "MCE":["Mathematics","ICT","Entrepreneurship"],
    "AEM":["Mathematics","Entrepreneurship","Economics"]
}

AHP_WEIGHTS = {
    "PCM":{"Mathematics":0.35,"Physics":0.35,"Chemistry":0.3},
    "PCB":{"Physics":0.35,"Chemistry":0.35,"Biology":0.3},
    "MPC":{"Mathematics":0.4,"Physics":0.35,"ICT":0.25},
    "MPG":{"Mathematics":0.4,"Physics":0.35,"Geography":0.25},
    "MCB":{"Mathematics":0.4,"Chemistry":0.35,"Biology":0.25},
    "BCG":{"Biology":0.4,"Chemistry":0.35,"Geography":0.25},
    "MEB":{"Mathematics":0.4,"Entrepreneurship":0.35,"Economics":0.25},
    "MEG":{"Mathematics":0.4,"Entrepreneurship":0.35,"Geography":0.25},
    "MCE":{"Mathematics":0.4,"ICT":0.35,"Entrepreneurship":0.25},
    "AEM":{"Mathematics":0.4,"Entrepreneurship":0.35,"Economics":0.25}
}

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def compute_subject_averages(row):
    # Only compute for subjects in SUBJECTS dict
    avg_dict = {}
    for subject, cols in SUBJECTS.items():
        # Check if all columns exist in the row
        existing_cols = [col for col in cols if col in row.index]
        if existing_cols:
            avg_dict[subject] = row[existing_cols].mean()
        else:
            avg_dict[subject] = 0
    return avg_dict

def eligible_combinations(avg):
    eligible = []
    for combo, subjects in RULES.items():
        if all(avg.get(s, 0) >= MIN_AVG for s in subjects):
            eligible.append(combo)
    return eligible

def topsis_score(avg, combo):
    weights = AHP_WEIGHTS[combo]
    values = np.array([avg.get(s, 0) for s in weights])
    w = np.array(list(weights.values()))
    norm = values / np.sqrt((values**2).sum())
    weighted = norm * w
    ideal_best = weighted.max()
    ideal_worst = weighted.min()
    d_pos = np.sqrt(((weighted-ideal_best)**2).sum())
    d_neg = np.sqrt(((weighted-ideal_worst)**2).sum())
    return d_neg / (d_pos + d_neg)

def create_manual_dataframe(form_data):
    """Create a DataFrame with ALL required columns from manual input"""
    if not ALL_SUBJECT_COLUMNS:
        # If we don't have feature list, create minimal DataFrame
        data = {}
        for subject, cols in SUBJECTS.items():
            for col in cols:
                val = form_data.get(col, 0)
                data[col] = [float(val) if val else 0]
        return pd.DataFrame(data)
    
    # Create DataFrame with all required features
    data = {}
    
    # Initialize all columns with 0
    for col in ALL_SUBJECT_COLUMNS:
        data[col] = [0]
    
    # Update with values from form
    for subject, cols in SUBJECTS.items():
        for col in cols:
            if col in ALL_SUBJECT_COLUMNS:  # Only update if column exists in model features
                val = form_data.get(col, 0)
                data[col] = [float(val) if val else 0]
    
    return pd.DataFrame(data)

def compute_learning_paths(df_input):
    df = df_input.copy()
    
    # Ensure all training features exist
    if features_list:
        # Add missing columns with default value 0
        for f in features_list:
            if f not in df.columns:
                df[f] = 0
        
        # Remove extra columns that weren't in training
        columns_to_keep = [col for col in df.columns if col in features_list or col.startswith(('Primary_', 'Alternative_', 'Confidence_', 'Decision_'))]
        df = df[columns_to_keep]
        
        X = df[features_list]
    else:
        # Fallback: use columns with '_y' suffix
        y_cols = [c for c in df.columns if "_y" in c]
        X = df[y_cols]
    
    # Make sure we have the right columns in the right order
    if features_list:
        X = X[features_list]
    
    proba = model.predict_proba(X)
    ml_preds = model.classes_[np.argmax(proba, axis=1)]
    ml_conf = np.max(proba, axis=1)

    primary, alternative, confidence, explanation = [], [], [], []

    for i, row in df.iterrows():
        avg = compute_subject_averages(row)
        eligible = eligible_combinations(avg)
        if not eligible:
            primary.append("NONE")
            alternative.append("NONE")
            confidence.append(0)
            explanation.append("No combination meets minimum requirements")
            continue

        scores = {c:topsis_score(avg,c) for c in eligible}
        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        topsis_primary = ranked[0][0]
        topsis_alt = ranked[1][0] if len(ranked)>1 else "NONE"

        # ML override
        if ml_conf[i]>=ML_OVERRIDE_THRESHOLD and ml_preds[i] in eligible:
            final_primary = ml_preds[i]
            reason = "ML override (high confidence)"
        else:
            final_primary = topsis_primary
            reason = "TOPSIS ranking"

        primary.append(final_primary)
        alternative.append(topsis_alt)
        confidence.append(round(ml_conf[i]*100,2))
        top_subjects = dict(sorted(avg.items(), key=lambda x:x[1], reverse=True)[:3])
        explanation.append(f"{reason}; Eligible={eligible}; Top subjects={top_subjects}")

    df["Primary_Learning_Path"] = primary
    df["Alternative_Learning_Path"] = alternative
    df["Confidence_%"] = confidence
    df["Decision_Explanation"] = explanation
    return df

# ======================================================
# ROUTES
# ======================================================
@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename=="":
        flash("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        if filename.endswith(".csv"):
            df_input = pd.read_csv(filepath)
        else:
            df_input = pd.read_excel(filepath)
            
        print(f"\nUploaded file columns ({len(df_input.columns)}):")
        for col in df_input.columns:
            print(f"  - {col}")

        result_df = compute_learning_paths(df_input)
        output_file = os.path.join(app.config["UPLOAD_FOLDER"], f"result_{filename}")
        result_df.to_csv(output_file,index=False)

        return render_template("results.html", tables=[result_df.to_html(classes='data',index=False)], filename=output_file)
    else:
        flash("File type not allowed")
        return redirect(request.url)

@app.route("/manual", methods=["GET","POST"])
def manual():
    if request.method=="POST":
        try:
            print("\nProcessing manual form data...")
            df_input = create_manual_dataframe(request.form)
            print(f"Created DataFrame with {len(df_input.columns)} columns")
            
            if features_list:
                missing = [col for col in features_list if col not in df_input.columns]
                extra = [col for col in df_input.columns if col not in features_list and col not in ['Primary_Learning_Path', 'Alternative_Learning_Path', 'Confidence_%', 'Decision_Explanation']]
                
                if missing:
                    print(f"WARNING: Missing columns: {missing}")
                if extra:
                    print(f"WARNING: Extra columns: {extra}")
            
            result_df = compute_learning_paths(df_input)
            return render_template("results.html", tables=[result_df.to_html(classes='data',index=False)], filename=None)
        except Exception as e:
            print(f"Error in manual processing: {str(e)}")
            flash(f"Error processing form: {str(e)}")
            return render_template("manual.html", subjects=SUBJECTS)
    
    # For GET request, still show only the main subjects
    return render_template("manual.html", subjects=SUBJECTS)

if __name__=="__main__":
    app.run(debug=True)