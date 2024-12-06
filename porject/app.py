from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('model_rrf.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the route for the home page (to show the form)
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction (after form submission)
@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from the form
    sub_area = int(request.form['sub_area'])
    property_type = int(request.form['property_type'])
    property_area = int(request.form['property_area'])
    company_name = int(request.form['company_name'])
    township_name = int(request.form['township_name'])
    clubhouse = int(request.form['clubhouse'])
    school = int(request.form['school'])
    hospital = int(request.form['hospital'])
    mall = int(request.form['mall'])
    park = int(request.form['park'])
    swimming_pool = int(request.form['swimming_pool'])
    gym = int(request.form['gym'])

    # Prepare the input array for the model
    input_data = np.array([[sub_area, property_type, property_area, company_name, township_name, 
                            clubhouse, school, hospital, mall, park, swimming_pool, gym]])

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result
    return render_template('index.html', prediction_text=f'predicted house price: ${round(prediction[0], 2)} M')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=0)
