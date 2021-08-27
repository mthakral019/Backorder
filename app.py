from flask import Flask,render_template,request,url_for
from flask import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import redirect
app= Flask(__name__)
model=pickle.load(open('easy_ensemble_model.pkl','rb'))

app=Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/success/<int:score>')
def success(score):
    res=""
    if score==1:
        res="YES BACKORDER"
    else:
        res="NO BACKORDER"
    exp={'score':score,'res':res}
    return render_template('result.html',result=exp)



def log_transform(a):
    sign =np.sign(a)
    a=np.log(1.0+abs(a))*sign
    return a

df_log=pd.read_csv('df_log.csv')
df_log.drop('went_on_backorder',axis=1,inplace=True)
sc=StandardScaler().fit(df_log)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        national_inv=float(request.form['national_inv'])
        national_inv=log_transform(national_inv)
        lead_time=float(request.form['lead_time'])
        in_transit_qty=float(request.form['in_transit_qty'])
        in_transit_qty=log_transform(in_transit_qty) 
        forecast_3_month=float(request.form['forecast_3_month'])
        forecast_6_month=float(request.form['forecast_6_month'])
        forecast_9_month=float(request.form['forecast_9_month'])
        forecast=(forecast_3_month+forecast_6_month+forecast_9_month)/3
        sales_1_month=float(request.form['sales_1_month'])
        sales_3_month=float(request.form['sales_3_month'])
        sales_6_month=float(request.form['sales_6_month'])
        sales_9_month=float(request.form['sales_9_month'])
        sales=(sales_1_month+ sales_3_month+ sales_6_month+sales_9_month)/4
        min_bank=float(request.form['min_bank'])
        min_bank=log_transform(min_bank)
        potential_issue=bool(request.form['potential_issue'])
        if(potential_issue=='Yes'):
            potential_issue=1
        else:
            potential_issue=0
        pieces_past_due=float(request.form['pieces_past_due'])
        pieces_past_due=log_transform(pieces_past_due)
        perf_6_month_avg=float(request.form['perf_6_month_avg'])
        perf_12_month_avg=float(request.form['perf_12_month_avg'])
        perf=(perf_6_month_avg+perf_12_month_avg)/2
        local_bo_qty=float(request.form['local_bo_qty'])
        local_bo_qty=log_transform(local_bo_qty)
        deck_risk=bool(request.form['deck_risk'])
        if(deck_risk=='Yes'):
            deck_risk=1
        else:
            deck_risk=0
        oe_constraint=bool(request.form['oe_constraint'])
        if(oe_constraint=='Yes'):
            oe_constraint=1
        else:
            oe_constraint=0
        ppap_risk=bool(request.form['ppap_risk'])
        if(ppap_risk=='Yes'):
            ppap_risk=1
        else:
            ppap_risk=0
        stop_auto_buy=bool(request.form['stop_auto_buy'])
        if(stop_auto_buy=='Yes'):
            stop_Auto_buy=1
        else:
            stop_auto_buy=0
        rev_stop=bool(request.form['rev_stop'])
        if(rev_stop=='Yes'):
            rev_stop=1
        else:
            rev_stop=0
        data=np.array([national_inv,lead_time,in_transit_qty,min_bank,potential_issue,pieces_past_due,
        local_bo_qty,deck_risk,oe_constraint,ppap_risk,stop_auto_buy,rev_stop,forecast,sales,perf]).reshape(1,-1)
        data=sc.transform(data)
        predictions=model.predict(data)

        return redirect(url_for('success',score=predictions))

    else:
        return render_template('index.html')    


if __name__ == "__main__" :
    app.run(debug=True)