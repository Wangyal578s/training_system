from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ------------------------
# Load Models and Metadata
# ------------------------
effect_model = joblib.load("models/effectiveness_model.pkl")
cost_model = joblib.load("models/cost_model.pkl")
training_meta = pd.read_pickle("models/training_metadata.pkl")


# ------------------------
# Recommendation Function
# ------------------------
def recommend_trainings(employee_profile, top_n=5):
    rows = []

    for _, t in training_meta.iterrows():
        row = {
            "grade": employee_profile["grade"],
            "designation": employee_profile["designation"],
            "section_branch": employee_profile["section_branch"],
            "division": employee_profile["division"],
            "department": employee_profile["department"],
            "employee_group_type": employee_profile["employee_group_type"],
            "training_type": t["training_type"],
            "training_category": t["training_category"],
            "development_type": t["development_type"],
            "training_country": t["training_country"],
            "duration_days": t["duration_days"],
            "duration_hrs.": t["duration_hrs."],
        }
        rows.append(row)

    X_eff = pd.DataFrame(rows)

    # Predict effectiveness probability (for high impact)
    probs = effect_model.predict_proba(X_eff)
    class_labels = effect_model.named_steps["model"].classes_

    if 2 in class_labels:
        idx_high = list(class_labels).index(2)
    else:
        idx_high = np.argmax(class_labels)

    p_high = probs[:, idx_high]

    # Predict cost
    X_cost = training_meta[[
        "training_type",
        "training_category",
        "development_type",
        "training_country",
        "training_institute",
        "duration_days",
        "duration_hrs."
    ]].copy()

    predicted_cost = cost_model.predict(X_cost)
    predicted_cost = np.clip(predicted_cost, 1e-6, None)

    # Score = effectiveness / cost
    score = p_high / predicted_cost

    results = training_meta.copy()
    results["predicted_effectiveness_prob"] = p_high
    results["predicted_cost"] = predicted_cost
    results["score"] = score

    results = results.sort_values("score", ascending=False).head(top_n)

    return results.reset_index(drop=True)


# ------------------------
# Routes
# ------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ===============================
# Recommendation Page + Handler
# ===============================
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return render_template("recommend.html")

    if request.method == "POST":
        employee_profile = {
            "grade": request.form.get("grade", "").lower(),
            "designation": request.form.get("designation", "").lower(),
            "section_branch": request.form.get("section_branch", "").lower(),
            "division": request.form.get("division", "").lower(),
            "department": request.form.get("department", "").lower(),
            "employee_group_type": request.form.get("employee_group_type", "").lower(),
        }

        df_results = recommend_trainings(employee_profile, top_n=5)
        results = df_results.to_dict(orient="records")

        return render_template("recommend.html", results=results)


# ===============================
# AJAX Cost Prediction Endpoint
# ===============================
@app.route("/predict_cost", methods=["POST"])
def predict_cost():
    try:
        data = {
            "training_type": request.form["training_type"].lower(),
            "training_category": request.form["training_category"].lower(),
            "development_type": request.form["development_type"].lower(),
            "training_country": request.form["training_country"].lower(),
            "training_institute": request.form["training_institute"].lower(),
            "duration_days": float(request.form["duration_days"]),
            "duration_hrs.": float(request.form["duration_hrs."]),
        }

        X = pd.DataFrame([data])
        prediction = cost_model.predict(X)[0]
        prediction = max(prediction, 0)

        return jsonify({
            "status": "success",
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ===============================
# Cost Prediction Page
# ===============================
@app.route("/cost", methods=["GET"])
def cost():
    return render_template("cost.html")


# ------------------------
# Run Application
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
