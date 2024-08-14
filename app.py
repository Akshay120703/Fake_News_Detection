from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Fake_News.html')  # Update the filename here

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        text = request.form['text']
        
        # Combine the title, author, and text
        data = f"{title} {author} {text}"
        
        # Vectorize the input data
        vect_data = vectorizer.transform([data])
        
        # Predict using the loaded model
        prediction = model.predict(vect_data)
        
        # Return the result
        if prediction[0] == 0:
            result = "The News is not fake!"
        else:
            result = "The News is fake!"
        
        return render_template('Fake_News.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
