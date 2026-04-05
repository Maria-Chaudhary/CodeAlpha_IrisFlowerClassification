from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io, json
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# ── Train model once at startup (same logic as iris_classifier.py) ──
iris = load_iris()
X = pd.DataFrame(iris.data, columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
y = pd.Series(iris.target_names[iris.target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# ── Helper: plot → base64 string ──
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Confusion matrix image ──
def make_cm():
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=model.classes_, yticklabels=model.classes_,
                ax=ax, linewidths=0.5, linecolor='#e0e0e0',
                annot_kws={'size': 15, 'weight': 'bold'})
    ax.set_facecolor('none')
    ax.set_xlabel('Predicted', color='#555', fontsize=11)
    ax.set_ylabel('Actual',    color='#555', fontsize=11)
    ax.tick_params(colors='#555')
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64

# ── Feature importance image ──
def make_fi():
    importances = model.feature_importances_
    features    = X.columns
    idx         = importances.argsort()[::-1]
    colors      = ['#2ecc71','#3498db','#e67e22','#95a5a6']
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_alpha(0)
    bars = ax.bar(features[idx], importances[idx], color=[colors[i] for i in idx],
                  width=0.5, zorder=3)
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#333')
    ax.set_facecolor('none')
    ax.set_ylabel('Importance Score', color='#555', fontsize=10)
    ax.tick_params(colors='#555', axis='both')
    ax.set_ylim(0, max(importances) + 0.06)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64

# ── Classification report as dict ──
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_rows = []
for cls in model.classes_:
    r = report_dict[cls]
    report_rows.append({
        'species': cls,
        'precision': round(r['precision'], 2),
        'recall':    round(r['recall'],    2),
        'f1':        round(r['f1-score'],  2),
        'support':   int(r['support'])
    })

# ── Feature importance list ──
importances = model.feature_importances_
idx         = importances.argsort()[::-1]
fi_list     = [{'feature': X.columns[i], 'score': round(float(importances[i]), 4)} for i in idx]

limits = {
    'SepalLengthCm': (4.0, 8.5),
    'SepalWidthCm' : (1.5, 5.0),
    'PetalLengthCm': (0.5, 7.5),
    'PetalWidthCm' : (0.1, 3.0),
}

@app.route('/')
def index():
    return render_template('index.html',
        accuracy   = accuracy,
        train_size = len(X_train),
        test_size  = len(X_test),
        total      = len(X),
        cm_img     = make_cm(),
        fi_img     = make_fi(),
        report     = report_rows,
        fi_list    = fi_list,
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    errors = {}
    values = {}
    for feat, (lo, hi) in limits.items():
        try:
            v = float(data.get(feat, ''))
            if not (lo <= v <= hi):
                errors[feat] = f'Must be between {lo} and {hi}'
            else:
                values[feat] = v
        except (ValueError, TypeError):
            errors[feat] = 'Enter a valid number'

    if errors:
        return jsonify({'success': False, 'errors': errors})

    new_flower   = pd.DataFrame([values], columns=X.columns)
    prediction   = model.predict(new_flower)[0]
    probs        = model.predict_proba(new_flower)[0]
    confidences  = {cls: round(float(p)*100, 1) for cls, p in zip(model.classes_, probs)}

    return jsonify({'success': True, 'prediction': prediction, 'confidences': confidences})

if __name__ == '__main__':
    app.run(debug=True)