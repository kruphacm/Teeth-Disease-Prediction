import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup
app = Flask(__name__)
model = pickle.load(open('modelteeth.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLteeth.html')

@app.route('/predict',methods=['POST'])
def predict():
    res1={'gum disease':1.0,'a cracked tooth':2.0,'worn-down fillings or crowns':3.0,'Black, white, or brown tooth stains':4.0,'Holes or pits in your teeth':5.0,"Pain when you bite down":6.0,'VYellowish discoloration':7.0,'Cracked or chipped teeth':8.0,'Grooves on your teeth’s surface':9.0,'bleeding':10.0,'pain':11.0,'sore throat':12.0,'Ear Pain':13.0,'Dramatic weight loss':14.0,'Difficulty chewing or swallowing':15.0,'Bad breath':16.0,'Painful chewing':17.0,'Red and swollen gums':18.0,'Tender or bleeding gums':19.0}
    int_features = [res1[x] for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'<p>Predicted Disease is dentist hypersensitivity<br><br>Treatment:mouthwash</p>',1:'<p>Predicted Disease is Cavity<br><br>Treatment:Brush your teeth with warm water</p>',2:'<p>Predicted Disease is Tooth Erosion<br><br>Treatment:Chew sugar-free gum</p>',3:'<p>Predicted Disease is Mouth Sores<br><br>Treatment:Cryotherapy</p>',4:'<p>Predicted Disease is Oral Cancer<br><br>Treatment:Eat a well balanced diet and Surgery</p>',5:'<p>Predicted Disease is Periodontitis<br><br>Treatment:Tooth polishing</p>'}
    output=res[output]
    output=Markup(output)
    return render_template('HTMLteeth.html', prediction_text=output)
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
