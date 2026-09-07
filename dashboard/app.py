import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from sample_data import generate_spatial_data, generate_timeseries_data, generate_model_results, generate_alerts

app = Flask(__name__)
CORS(app)

# Generate sample data on startup
spatial_data = generate_spatial_data()
model_results = generate_model_results()
alerts_data = generate_alerts()
# timeseries data will be generated on demand per zone

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --- APIs ---
@app.route('/api/drought-data')
def api_drought_data():
    return jsonify(spatial_data)

@app.route('/api/timeseries/<zone>')
def api_timeseries(zone):
    data = generate_timeseries_data(zone)
    return jsonify(data)

@app.route('/api/alerts')
def api_alerts_data():
    return jsonify(alerts_data)

@app.route('/api/model-results')
def api_model_results():
    # Check if real trained model evaluation report exists from Colab
    import json
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_candidates = [
        os.path.join(project_root, "outputs", "reports", "final_comparison.json"),
        os.path.join(project_root, "outputs", "reports", "model_comparison.json"),
    ]
    for rpath in report_candidates:
        if os.path.exists(rpath):
            try:
                with open(rpath, 'r', encoding='utf-8') as f:
                    real_results = json.load(f)
                    base = generate_model_results()
                    if 'comparison' in real_results:
                        base['comparison'] = real_results['comparison']
                    if 'feature_importance' in real_results:
                        base['feature_importance'] = real_results['feature_importance']
                    if 'lead_time_skill' in real_results:
                        base['lead_time_skill'] = real_results['lead_time_skill']
                    return jsonify(base)
            except Exception:
                pass
    return jsonify(model_results)

if __name__ == '__main__':
    # Run standalone
    app.run(debug=True, host='0.0.0.0', port=5000)
