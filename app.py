from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# 🔥 Load all trained objects
model = pickle.load(open('restaurent_predictor.pkl', 'rb'))
mlb = pickle.load(open('multilabel_binarizer.pkl', 'rb'))
ohe = pickle.load(open('onehotencoder.pkl', 'rb'))
scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
model_columns = pickle.load(open('columns.pkl', 'rb'))


# 🔹 City → Tier mapping
tier1 = ['Mumbai', 'Kolkata', 'Bangalore','Chennai']
tier2 = ['Pune', 'Hyderabad', 'Kochi', 'Trivandrum','Mysore','Surat','Lucknow']


# 🔥 GenAI-style suggestion function
def generate_suggestions(input_data, predicted_rating):
    suggestions = []

    if input_data['has_online_delivery'] == 0:
        suggestions.append("Enable online delivery to reach more customers")

    if input_data['has_table_booking'] == 0:
        suggestions.append("Add table booking option for better customer convenience")

    if input_data['cuisine_count'] < 3:
        suggestions.append("Increase cuisine variety to attract more customers")

    if input_data['area_popularity'] < 0.5:
        suggestions.append("Improve marketing or visibility in your area")

    if input_data['cost_numeric'] > 1200:
        suggestions.append("Consider optimizing pricing strategy")

    if predicted_rating < 3.5:
        suggestions.append("Focus on improving service quality and customer experience")

    if predicted_rating > 4.7:
        suggestions.append("Great performance! Maintain consistency and customer satisfaction")

    if len(suggestions) == 0:
        suggestions.append("Your restaurant setup looks well balanced 👍")

    return suggestions


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------- INPUT --------
        status = int(request.form['status'])
        online = int(request.form['online'])
        table = int(request.form['table'])
        area_popularity = float(request.form['area_popularity'])
        cost = float(request.form['cost'])
        cost_cat = int(request.form['cost_category'])

        city = request.form['city']

        # City → Tier
        if city in tier1:
            city_tier = 1
        elif city in tier2:
            city_tier = 2
        else:
            city_tier = 3

        cuisines = request.form.getlist('cuisine')
        cuisine_count = len(cuisines)

        r_type = request.form['restaurant_type']

        # -------- BASE DATAFRAME --------
        df_input = pd.DataFrame({
            'status': [status],
            'has_online_delivery': [online],
            'has_table_booking': [table],
            'cuisine_count': [cuisine_count],
            'area_popularity': [area_popularity],
            'cost_numeric': [cost],
            'city_tier': [city_tier],
            'cost_category_encoded': [cost_cat],
            'restaurant_type': [r_type]
        })

        # -------- ENCODING --------

        # MultiLabelBinarizer
        cuisine_encoded = mlb.transform([cuisines])
        cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_)

        # OneHotEncoder
        rtype_encoded = ohe.transform(df_input[['restaurant_type']])
        rtype_df = pd.DataFrame(
            rtype_encoded,
            columns=ohe.get_feature_names_out(['restaurant_type'])
        )

        # Combine all
        df_final = pd.concat(
            [df_input.drop('restaurant_type', axis=1),
             cuisine_df,
             rtype_df],
            axis=1
        )

        # -------- COLUMN ALIGNMENT --------
        for col in model_columns:
            if col not in df_final.columns:
                df_final[col] = 0

        df_final = df_final[model_columns]

        # -------- SCALING --------
        df_scaled = scaler.transform(df_final)

        # -------- PREDICTION --------
        prediction = model.predict(df_scaled)[0]
        prediction = round(prediction, 2)

        # -------- GEN AI SUGGESTIONS --------
        suggestions = generate_suggestions(df_input.iloc[0], prediction)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Rating: {prediction} ⭐",
            suggestions=suggestions
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="Error: Please check input values",
            suggestions=[]
        )


if __name__ == "__main__":
    app.run(debug=True)