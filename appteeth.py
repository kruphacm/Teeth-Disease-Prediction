import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelteeth.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLteeth.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x)-1 for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'dentin hypersensitivity Treatment:mouthwash',1:'Cavity    Treatment:Brush your teeth with warm water',2:'Tooth Erosion   Treatment:Chew sugar-free gum',3:'Mouth Sores    Treatment:Cryotherapy',4:'Oral Cancer    Treatment:Eat a well balanced diet and Surgery',5:'Periodontitis    Treatment:Tooth polishing'}
    output=res[output]

    return render_template('HTMLteeth.html', prediction_text='Predicted Disease is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)