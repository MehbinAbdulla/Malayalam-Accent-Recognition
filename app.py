from flask import Flask, render_template, request, jsonify
import os
import joblib
from train import train_model, extract_features

app = Flask(__name__)
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_accent():
    try:
        audio_file = request.files.get('audio_file')

        if not audio_file:
            app.logger.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400

        save_path = os.path.join("uploads", audio_file.filename)
        app.logger.info(f"Saving audio file to {save_path}")
        
        try:
            audio_file.save(save_path)
        except Exception as e:
            app.logger.error(f"Error saving audio file: {e}")
            return jsonify({"error": str(e)}), 500

        try:
            result = predict_accent_function(save_path)
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500

        return render_template('result.html', accent=result[0], prob_kkd=result[1]*100, prob_tsr=result[2]*100, audio=audio_file.filename)
    
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/nav')
def navigate():
    return render_template("index.html")

def predict_accent_function(audio_path):
    global model
    try:
        if model is None:
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
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True, host='0.0.0.0')