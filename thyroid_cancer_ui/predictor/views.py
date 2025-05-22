from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pandas as pd
import os
from .forms import ThyroidCancerForm

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'thyroid_cancer_model.pkl')
model = joblib.load(model_path)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Map fields to encoder filenames
encoder_file_map = {
    'Gender': 'Gender_encoder.pkl',
    'Smoking': 'Smoking_encoder.pkl',
    'Hx Smoking': 'Hx Smoking_encoder.pkl',
    'Hx Radiothreapy': 'Hx Radiotherapy_encoder.pkl',  # Spelled as in training
    'M': 'M_encoder.pkl',  # Was "Metastasis"
    'Physical Examination': 'Physical_Examination_encoder.pkl',
    'Adenopathy': 'Adenopathy_encoder.pkl',
    'Pathology': 'M_encoder.pkl',
    'Focality': 'Focality_encoder.pkl',
    'Risk': 'M_encoder.pkl',
    'N': 'M_encoder.pkl',  # Was "Node Status"
    'T': 'M_encoder.pkl',  # Was "Tumor Size"
    'Stage': 'M_encoder.pkl',
    'Response': 'M_encoder.pkl',
}

def safe_transform(encoder, series):
    known_classes = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)

def predict_recurrence(request):
    if request.method == 'POST':
        form = ThyroidCancerForm(request.POST)
        if form.is_valid():
            try:
                # Get form data and map fields to match model training
                data = {
                    'Age': [form.cleaned_data['age']],
                    'Gender': [form.cleaned_data['gender']],
                    'Smoking': [form.cleaned_data['smoking']],
                    'Hx Smoking': [form.cleaned_data['hx_smoking']],
                    'Hx Radiothreapy': [form.cleaned_data['hx_radiotherapy']],  # Fix spelling
                    'Thyroid Function': [form.cleaned_data['thyroid_function']],
                    'Physical Examination': [form.cleaned_data['physical_examination']],
                    'Adenopathy': [form.cleaned_data['adenopathy']],
                    'Pathology': [form.cleaned_data['pathology']],
                    'Focality': [form.cleaned_data['focality']],
                    'Risk': [form.cleaned_data['risk']],
                    'T': [form.cleaned_data['tumor_size']],  # Rename
                    'N': [form.cleaned_data['node_status']],  # Rename
                    'M': [form.cleaned_data['metastasis']],  # Rename
                    'Stage': [form.cleaned_data['stage']],
                    'Response': [form.cleaned_data['response']],
                }

                input_df = pd.DataFrame(data)

                # Encode categorical features
                for col in input_df.select_dtypes(include='object').columns:
                    encoder_filename = encoder_file_map.get(col)
                    if not encoder_filename:
                        input_df[col] = -1
                        continue

                    encoder_path = os.path.join(BASE_DIR, 'predictor', encoder_filename)
                    if not os.path.exists(encoder_path):
                        input_df[col] = -1
                        continue

                    encoder = joblib.load(encoder_path)
                    input_df[col] = safe_transform(encoder, input_df[col].astype(str))

                # Predict
                prediction = model.predict(input_df)[0]
                result = "Yes" if prediction == 1 else "No"
                return JsonResponse({'prediction': result})

            except Exception as e:
                return JsonResponse({'error': str(e)})
        return JsonResponse({'error': 'Invalid form data'})
    else:
        form = ThyroidCancerForm()
    return render(request, 'predictor/index.html', {'form': form})
