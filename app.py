from flask import Flask, render_template, request, jsonify
import os
from train import *

app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_accent():
  audio_file = request.files['audio_file']

  if audio_file:
    result=["null"]
    if audio_file.filename is None:           
      # Handle the case when filename is None
      return jsonify({"error": "Filename is None"})
      
    save_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(save_path)
    result = predict_accent_function(save_path)    
    return render_template('result.html',accent=result[0], prob_kkd=result[1]*100, prob_tsr=result[2]*100, audio=audio_file.filename)
  else:
    return render_template('result.html',accent="No Results Found", prob_kkd="0", prob_tsr="0", audio=audio_file.filename)

@app.route('/nav')
def navigate():
  return render_template("index.html")

def predict_accent_function(audio_path):
  if os.path.exists('train/trained_accent_model.pkl'):
    model_path = 'train/trained_accent_model.pkl'
    model = joblib.load(model_path)
  else:
    train_model()
    model_path = 'train/trained_accent_model.pkl'
    model = joblib.load(model_path)

  new_audio_features = extract_features(audio_path)
  new_audio_features_reshaped = new_audio_features.reshape(1, -1)
  predicted_probabilities = model.predict_proba(new_audio_features_reshaped)
  prob = predicted_probabilities[0]
  predicted_accent = model.predict(new_audio_features_reshaped)[0]
  accent_label = 'Kozhikode Slang' if predicted_accent == 0 else 'Trissur Slang'
  return_data = (accent_label, prob[0], prob[1])
  return return_data


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
