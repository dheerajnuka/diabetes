from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3
import pandas
# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        conn = sqlite3.connect('test.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS Daibetes 
           (preg               INT    NOT NULL,
            glucose            INT     NOT NULL,
            bp                 INT     NOT NULL,
            st                 INT     NOT NULL,
            insulin            INT     NOT NULL,
            bmi                INT     NOT NULL,
            dpf                INT     NOT NULL,
            age                INT     NOT NULL);''')

        print("Table created successfully")
        print("Operation done successfully")
        conn = sqlite3.connect('test.db')
        print("Opened database successfully")
        conn.execute("INSERT INTO Daibetes (preg,glucose,bp,st,insulin,bmi,dpf,age) VALUES (?,?,?,?,?,?,?,?)",(preg,glucose,bp,st,insulin,bmi,dpf,age))
        conn.commit()
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
