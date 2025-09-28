import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load models
model_td = pickle.load(open('randomforest.pkl', 'rb'))       # Thủ Đức
model_th = pickle.load(open('randomforestTH.pkl', 'rb'))     # Tân Hiệp


@app.route('/')
def home():
    return render_template('index.html')


# Predict Thủ Đức
@app.route('/predict_td', methods=['POST'])
def predict_td():
    nhietDo = request.form['nhietDo']
    pH = request.form['pH']
    doDuc = request.form['doDuc']
    doMau = request.form['doMau']
    chatLoLung = request.form['chatLoLung']
    doDan = request.form['doDan']

    int_features = [float(nhietDo), float(pH), float(doDuc), float(doMau), float(chatLoLung), float(doDan)]
    final_features = [np.array(int_features)]
    prediction = model_td.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text_td=f'{output:.2f} mg/L', active_tab="td")


# Predict Tân Hiệp
@app.route('/predict_th', methods=['POST'])
def predict_th():
    pH = request.form['pH_TH']
    doDuc = request.form['doDuc_TH']
    doMau = request.form['doMau_TH']
    chatLoLung = request.form['chatLoLung_TH']
    doDan = request.form['doDan_TH']

    int_features = [float(pH), float(doDuc), float(doMau), float(chatLoLung), float(doDan)]
    final_features = [np.array(int_features)]
    prediction = model_th.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text_th=f'{output:.2f} mg/L', active_tab="th")


# Predict EC => Mặn
@app.route('/predict_ec', methods=['POST'])
def predict_ec():
    try:
        ec_value = float(request.form['ec'])
        option = request.form['option']

        if option == "thuduc":
            result = ((-2e-6)*ec_value**3)+(0.0015*ec_value**2)-(0.0208*ec_value)+1.9075
        elif option == "tanhiep":
            result = (0.0002*ec_value*ec_value)+(0.0794*ec_value)+5.3813
        else:
            result = "Chưa chọn nhà máy"
    except:
        result = "Lỗi nhập dữ liệu"

    return render_template('index.html', result=f'{result:.2f} mg/L', active_tab="ec")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
