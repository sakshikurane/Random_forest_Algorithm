import os
import pickle
 
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
 
app = Flask(__name__)
 
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "randomforest.pkl")
with open(MODEL_PATH, "rb") as f:  # only load pickles you trust
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)
 
 
def codes(label, n):
    """Placeholder labels 'Label 0' .. 'Label n-1'. Replace with real names."""
    return [(i, f"{label} {i}") for i in range(n)]
 
 
# ----------------------------------------------------------------------------
# EDIT ME: code -> display label for every label-encoded column.
# Sizes are the minimum implied by the model's split thresholds.
# ----------------------------------------------------------------------------
CATEGORIES = {
    "company_name": codes("Company", 20),
    "industry": codes("Industry", 7),
    "country": codes("Country", 6),
    "company_size": codes("Size", 4),
    "month": [
        (i, m)
        for i, m in enumerate(
            ["April", "August", "December", "February", "January", "July",
             "June", "March", "May", "November", "October", "September"]
        )
    ],  # alphabetical = LabelEncoder order; change if you encoded differently
    "reason_for_layoffs": codes("Reason", 5),
    "hiring_trend": codes("Trend", 4),
    "top_hiring_role": codes("Role", 8),
}
 
# Numeric inputs: (label, min, max, step, default, hint)
NUMERIC = {
    "year": ("Year", 2024, 2025, 1, 2025, "Reporting year"),
    "layoffs_count": ("Layoffs count", 0, 15000, 1, 2400, "Employees laid off"),
    "layoff_percentage": ("Layoff %", 0, 30, 0.1, 9, "Share of workforce"),
    "ai_automation_impact": ("AI automation impact", 0, 20, 0.1, 6, "Estimated % of roles affected"),
    "ai_replacement_risk": ("AI replacement risk", 1, 10, 0.1, 6, "Scale 1 to 10"),
    "open_roles": ("Open roles", 0, 10000, 1, 2200, "Currently advertised"),
    "remote_jobs_percentage": ("Remote jobs %", 0, 100, 1, 50, "Share of remote roles"),
    "stock_growth_percent": ("Stock growth %", -50, 90, 0.1, 22, "Year over year"),
    "revenue_growth_percent": ("Revenue growth %", -30, 60, 0.1, 16, "Year over year"),
    "salary_budget_change": ("Salary budget change %", -20, 35, 0.1, 7, "Year over year"),
    "ai_adoption_level": ("AI adoption level", 1, 10, 0.1, 6, "Scale 1 to 10"),
    "employee_sentiment": ("Employee sentiment", 1, 10, 0.1, 7, "Scale 1 to 10"),
    "job_security_score": ("Job security score", 1, 10, 0.1, 6, "Scale 1 to 10"),
}
 
CATEGORY_LABELS = {
    "company_name": "Company",
    "industry": "Industry",
    "country": "Country",
    "company_size": "Company size",
    "month": "Month",
    "reason_for_layoffs": "Reason for layoffs",
    "hiring_trend": "Hiring trend",
    "top_hiring_role": "Top hiring role",
}
 
# Form sections: (title, description, [fields])
SECTIONS = [
    ("Company profile", "Who and where",
     ["company_name", "industry", "country", "company_size", "month", "year"]),
    ("Layoffs", "Size and cause of workforce cuts",
     ["layoffs_count", "layoff_percentage", "reason_for_layoffs"]),
    ("Hiring", "Demand for talent",
     ["open_roles", "hiring_trend", "top_hiring_role", "remote_jobs_percentage"]),
    ("Financials", "Company performance",
     ["stock_growth_percent", "revenue_growth_percent", "salary_budget_change"]),
    ("AI and people", "Automation and workforce mood",
     ["ai_automation_impact", "ai_replacement_risk", "ai_adoption_level",
      "employee_sentiment", "job_security_score"]),
]
 
OUTCOMES = {
    "Bull Market": {"color": "#12997f", "note": "Growth signals dominate: hiring and financials point upward."},
    "Recession": {"color": "#d6455a", "note": "Cuts and weak financials point to a downturn."},
    "Stable": {"color": "#3d5a9e", "note": "Mixed signals: no strong move in either direction."},
}
 
PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Market Outlook</title>
<style>
  :root {
    --navy: #14213d;
    --navy-2: #1f3158;
    --amber: #fca311;
    --amber-d: #e08e00;
    --bg: #f2f5fa;
    --card: #ffffff;
    --ink: #1b2538;
    --muted: #66728a;
    --line: #dde3ee;
    --bull: #12997f;
    --rec: #d6455a;
    --stable: #3d5a9e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.5;
  }
  header {
    background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 100%);
    color: #fff; padding: 36px 24px 90px;
  }
  .wrap { max-width: 1120px; margin: 0 auto; }
  header h1 { margin: 0 0 6px; font-size: 2rem; letter-spacing: .2px; }
  header h1 span { color: var(--amber); }
  header p { margin: 0; color: #c3cde3; max-width: 60ch; }
  main { margin-top: -60px; padding: 0 24px 48px; }
  .layout { display: grid; grid-template-columns: 1fr 340px; gap: 24px; align-items: start; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px 24px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(20,33,61,.07);
  }
  .card h2 { margin: 0; font-size: 1.1rem; color: var(--navy); }
  .card .sub { margin: 2px 0 16px; color: var(--muted); font-size: .88rem; }
  .card h2::before {
    content: ""; display: inline-block; width: 4px; height: 1em; margin-right: 10px;
    background: var(--amber); border-radius: 2px; vertical-align: -2px;
  }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px 18px; }
  label { display: block; font-size: .84rem; font-weight: 600; color: var(--navy); margin-bottom: 4px; }
  label small { display: block; font-weight: 400; color: var(--muted); }
  input, select {
    width: 100%; padding: 10px 12px; font: inherit; color: var(--ink);
    background: #f9fbfe; border: 1px solid var(--line); border-radius: 9px;
  }
  input:focus, select:focus, button:focus-visible {
    outline: 3px solid rgba(252,163,17,.45); border-color: var(--amber);
  }
  .actions { display: flex; gap: 12px; flex-wrap: wrap; }
  button {
    font: inherit; font-weight: 700; border: 0; border-radius: 10px; padding: 13px 26px;
    cursor: pointer; transition: background .15s, transform .05s;
  }
  button:active { transform: translateY(1px); }
  .primary { background: var(--amber); color: var(--navy); }
  .primary:hover { background: var(--amber-d); }
  .ghost { background: #e8edf6; color: var(--navy); }
  .ghost:hover { background: #dbe3f1; }
  aside { position: sticky; top: 16px; }
  .result-empty { color: var(--muted); font-size: .95rem; }
  .verdict { border-radius: 12px; padding: 18px; color: #fff; margin-bottom: 18px; }
  .verdict .name { font-size: 1.7rem; font-weight: 800; line-height: 1.1; }
  .verdict .conf { opacity: .9; margin-top: 4px; }
  .verdict .note { margin-top: 10px; font-size: .9rem; opacity: .95; }
  .bar-row { margin-bottom: 12px; }
  .bar-top { display: flex; justify-content: space-between; font-size: .88rem; font-weight: 600; }
  .track { background: #e8edf6; border-radius: 99px; height: 10px; overflow: hidden; margin-top: 4px; }
  .fill { height: 100%; border-radius: 99px; width: 0; transition: width .5s ease; }
  .err { background: #fdecef; color: #9b2335; border-radius: 10px; padding: 12px 14px; font-size: .9rem; }
  .imp { font-size: .84rem; color: var(--muted); }
  .imp b { color: var(--navy); }
  footer { text-align: center; color: var(--muted); font-size: .8rem; padding-bottom: 28px; }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    aside { position: static; order: -1; }
    header h1 { font-size: 1.6rem; }
  }
  @media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Job Market <span>Outlook</span></h1>
    <p>Enter company, layoff, hiring and financial signals. The random forest model classifies the outlook as Bull Market, Recession or Stable.</p>
  </div>
</header>
 
<main>
  <div class="wrap layout">
    <form id="form" autocomplete="off">
      {% for title, desc, fields in sections %}
      <section class="card">
        <h2>{{ title }}</h2>
        <p class="sub">{{ desc }}</p>
        <div class="grid">
          {% for f in fields %}
            {% if f in categories %}
              <div>
                <label for="{{ f }}">{{ cat_labels[f] }}</label>
                <select id="{{ f }}" name="{{ f }}">
                  {% for code, name in categories[f] %}
                  <option value="{{ code }}">{{ name }}</option>
                  {% endfor %}
                </select>
              </div>
            {% else %}
              {% set n = numeric[f] %}
              <div>
                <label for="{{ f }}">{{ n[0] }}<small>{{ n[5] }}</small></label>
                <input type="number" id="{{ f }}" name="{{ f }}"
                       min="{{ n[1] }}" max="{{ n[2] }}" step="{{ n[3] }}" value="{{ n[4] }}" required>
              </div>
            {% endif %}
          {% endfor %}
        </div>
      </section>
      {% endfor %}
      <div class="actions">
        <button type="submit" class="primary">Predict outlook</button>
        <button type="reset" class="ghost" id="reset">Reset values</button>
      </div>
    </form>
 
    <aside>
      <section class="card" aria-live="polite">
        <h2>Prediction</h2>
        <p class="sub">Result appears here</p>
        <div id="result"><p class="result-empty">Fill in the form and choose Predict outlook.</p></div>
      </section>
      <section class="card">
        <h2>Top drivers</h2>
        <p class="sub">What the model weighs most overall</p>
        <div class="imp">
          {% for name, val in importances %}
            <div class="bar-row">
              <div class="bar-top"><span>{{ name }}</span><span>{{ val }}%</span></div>
              <div class="track"><div class="fill" style="width: {{ val * 3 }}%; background: var(--navy-2)"></div></div>
            </div>
          {% endfor %}
        </div>
      </section>
    </aside>
  </div>
</main>
<footer>Model: RandomForestClassifier, {{ n_trees }} trees, {{ n_features }} features</footer>
 
<script>
const COLORS = {{ colors|tojson }};
const NOTES = {{ notes|tojson }};
const result = document.getElementById("result");
 
document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = Object.fromEntries(new FormData(e.target).entries());
  result.innerHTML = '<p class="result-empty">Predicting...</p>';
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    render(data);
  } catch (err) {
    result.innerHTML = '<div class="err">' + err.message + '</div>';
  }
});
 
function render(data) {
  const top = data.prediction;
  const pct = (data.probabilities[top] * 100).toFixed(1);
  let html = '<div class="verdict" style="background:' + COLORS[top] + '">' +
             '<div class="name">' + top + '</div>' +
             '<div class="conf">' + pct + '% of trees agree</div>' +
             '<div class="note">' + NOTES[top] + '</div></div>';
  Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([k, v]) => {
      const p = (v * 100).toFixed(1);
      html += '<div class="bar-row"><div class="bar-top"><span>' + k + '</span><span>' + p + '%</span></div>' +
              '<div class="track"><div class="fill" data-w="' + p + '" style="background:' + COLORS[k] + '"></div></div></div>';
    });
  result.innerHTML = html;
  requestAnimationFrame(() => result.querySelectorAll(".fill").forEach(el => el.style.width = el.dataset.w + "%"));
}
 
document.getElementById("reset").addEventListener("click", () => {
  result.innerHTML = '<p class="result-empty">Fill in the form and choose Predict outlook.</p>';
});
</script>
</body>
</html>
"""
 
 
def build_importances(top_n=6):
    pairs = sorted(zip(FEATURES, model.feature_importances_), key=lambda p: -p[1])[:top_n]
    return [(name.replace("_", " ").capitalize(), round(float(v) * 100, 1)) for name, v in pairs]
 
 
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        sections=SECTIONS,
        categories=CATEGORIES,
        cat_labels=CATEGORY_LABELS,
        numeric=NUMERIC,
        importances=build_importances(),
        colors={k: v["color"] for k, v in OUTCOMES.items()},
        notes={k: v["note"] for k, v in OUTCOMES.items()},
        n_trees=model.n_estimators,
        n_features=len(FEATURES),
    )
 
 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
    for feat in FEATURES:
        if feat not in data or str(data[feat]).strip() == "":
            return jsonify(error=f"Missing value for '{feat}'"), 400
        try:
            row[feat] = float(data[feat])
        except (TypeError, ValueError):
            return jsonify(error=f"'{feat}' must be a number"), 400
 
    X = pd.DataFrame([row], columns=FEATURES)  # keeps the training column order
    proba = model.predict_proba(X)[0]
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return jsonify(prediction=max(probs, key=probs.get), probabilities=probs)
 
 
if __name__ == "__main__":
    app.run(debug=True)