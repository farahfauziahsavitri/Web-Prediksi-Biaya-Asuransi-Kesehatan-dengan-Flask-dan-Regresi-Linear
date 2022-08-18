from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import statsmodels.api as sm

url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)
xnol = df[['age','bmi','children','smoker','region']]
df_x = sm.add_constant(xnol)
df_y = df['charges']
#penaksiran parameter
df_x_t = np.transpose(df_x)
df_x_dot = np.dot(df_x_t,df_x)
df_x_dot_inv = np.linalg.inv(df_x_dot)
df_x_dot_inv_dotx = np.dot(df_x_dot_inv,df_x_t)
B = np.dot(df_x_dot_inv_dotx,df_y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('rg/chart.html')


@app.route('/rg', methods = ['GET','POST'])
def regression():
    if request.method =='GET':
        return render_template('rg/input.html')
    else:
        age = request.form['Age']
        bmi = request.form['BMI']
        children = request.form['Number of children']
        smoker = request.form['Smoker (yes/no)']
        region = request.form['Region in US']

        def enc_smoker(p):
            if smoker == 'yes':
                return str(1)
            elif smoker == 'no' :
                return str(0)

        def enc_region (q):
            if region == 'northwest':
                return str(0)
            elif region == 'southeast' :
                return str(1)
            elif region == 'southwest':
                return str(2)
            elif region == 'northeast':
                return str(3)
        I = [1,int(age),int(bmi),int(children),int(enc_smoker(smoker)),int(enc_region(region))]
        y = np.dot(np.transpose(I),B)
        if y < 0 :
            return render_template('rg/output.html',pred = 0)
    return render_template('rg/output.html',pred = y)

if __name__ == '__main__':
    app.run(debug=True)