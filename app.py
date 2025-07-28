from flask import Flask, redirect,request,render_template,url_for
import pickle
import pandas as pd
import joblib

with open ("Razorpay_frauds.pkl","rb") as f:
    model=joblib.load(f)

with open ("label_encoders.pkl","rb") as f:
    label_encoder=joblib.load(f)

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def Fraud_detection():
     try:
        if request.method=="POST":
            User_ID=int(request.form["User_ID"])
            transaction_amount=float(request.form["Transaction_Amount"])
            transaction_type=request.form["Transaction_Type"]
            time_of_transaction=int(request.form["Time_of_Transaction"])
            device_used=request.form["Device_Used"]
            location=request.form["Location"]
            previous_fraudulent_transaction=int(request.form["Previous_Fraudulent_Transactions"])
            account_age=int(request.form["Account_Age"])
            number_of_transaction_last_24h=int(request.form["Number_of_Transactions_Last_24H"])
            payment_method=request.form["Payment_Method"]

            def safe_transform(le, value, col_name):
                if value not in le.classes_:
                    raise ValueError(f"'{value}' is not a known label in '{col_name}'. Known labels: {list(le.classes_)}")
                return le.transform([value])[0]

            transaction_type = safe_transform(label_encoder['Transaction_Type'], transaction_type, "Transaction_Type")
            device_used = safe_transform(label_encoder['Device_Used'], device_used, "Device_Used")
            location = safe_transform(label_encoder['Location'], location, "Location")
            payment_method = safe_transform(label_encoder['Payment_Method'], payment_method, "Payment_Method")


           
            input_data=pd.DataFrame({
                "User_ID":[User_ID],
                "Transaction_Amount":[transaction_amount],
                "Transaction_Type":[transaction_type],
                "Time_of_Transaction":[time_of_transaction],
                "Device_Used":[device_used],
                "Location":[location],
                "Previous_Fraudulent_Transactions":[previous_fraudulent_transaction],
                "Account_Age":[account_age],
                "Number_of_Transactions_Last_24H":[number_of_transaction_last_24h],
                "Payment_Method":[payment_method]
            })

        # Predict
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # Probability of class 1

        result = "Fraudulent" if prediction == 1 else "Legitimate"
        confidence = f"{prob*100:.2f}%"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

     except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

        

if __name__=="__main__":
    app.run(debug=True)




