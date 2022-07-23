import math

import pickle

import flask
import pandas as pd
from python_package import package
from scipy.stats import stats
from sklearn import preprocessing
import seaborn as sns
import xgboost

from flask import request, url_for, redirect, render_template
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')

app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

@app.route('/', methods=['GET', 'POST'])
def index():
    data = pd.read_csv('dataset/kr-final-cleaned.csv', low_memory=False)
    img = BytesIO()

    if flask.request.method == 'GET':
        return render_template('index.html', inf_mort_rate=package.infant_mortality_rate(data, img),
                               inf_mort_rate_yr=package.total_infant_mortality_year(data, img))

    if flask.request.method == 'POST':
        print("Predict")
        if request.form['btn'] == 'predict':
            v012 = float(flask.request.form['v012'])
            v729_label = (flask.request.form['v729_label'])
            v149_label = (flask.request.form['v149_label'])
            v150_label = (flask.request.form['v150_label'])
            v504_label = (flask.request.form['v504_label'])
            v501_one = (flask.request.form['v501_one'])
            v513_label = (flask.request.form['v513_label'])
            v203 = float(flask.request.form['v203'])
            v137 = float(flask.request.form['v137'])
            v202 = float(flask.request.form['v202'])
            v219 = float(flask.request.form['v219'])
            v201 = float(flask.request.form['v201'])
            v119_label = (flask.request.form['v119_label'])
            v122_label = (flask.request.form['v122_label'])
            v121_label = (flask.request.form['v121_label'])
            v116_label = (flask.request.form['v116_label'])
            bord = float(flask.request.form['bord'])
            m15_one = (flask.request.form['m15_one'])
            b2 = float(flask.request.form['b2'])
            m19 = float(flask.request.form['m19'])
            m3a_label = (flask.request.form['m3a_label'])
            m19a = (flask.request.form['m19a'])
            v602_one = (flask.request.form['v602_one'])
            v626_one = (flask.request.form['v626_one'])
            v614 = float(flask.request.form['v614'])
            v312_one = (flask.request.form['v312_one'])
            v364_one = (flask.request.form['v364_one'])
            v337 = float(flask.request.form['v337'])
            v362_one = (flask.request.form['v362_one'])
            v361_one = (flask.request.form['v361_one'])
            v327_one = (flask.request.form['v327_one'])
            v313_one = (flask.request.form['v313_one'])

            # print(str(v012))
            # print(v729_label)
            # print(v149_label)
            # print(v150_label)
            # print(v504_label)
            # print(v501_one)
            # print(v513_label)
            # print(v203)
            # print(v137)
            # print(v202)
            # print(v219)
            # print(v201)
            # print(v119_label)
            # print(v122_label)
            # print(v121_label)
            # print(v116_label)
            # print(bord)
            # print(m15_one)
            # print(b2)
            # print(m19)
            # print(m3a_label)
            # print(v602_one)
            # print(v012)
            # print(m19a)
            # print(v626_one)
            # print(v614)
            # print(v312_one)
            # print(v364_one)
            # print(v362_one)
            # print(v337)
            # print(v363_one)
            # print(v361_one)
            # print(v327_one)
            # print(v313_one)

            v729_encoded = 0
            if v729_label == "No education":
                v729_encoded = 0
            elif v729_label == "Incomplete primary":
                v729_encoded = 1
            elif v729_label == "Complete primary":
                v729_encoded = 2
            elif v729_label == "Incomplete secondary":
                v729_encoded = 3
            elif v729_label == "Complete secondary":
                v729_encoded = 4
            elif v729_label == "Higher":
                v729_encoded = 5
            elif v729_label == "Don't know":
                v729_encoded = 7
            else:
                v729_encoded = 9
            v149_encoded = 0
            print("hakdog")
            if v149_label == "No education":
                v149_encoded = 0
            elif v149_label == "Incomplete primary":
                v149_encoded = 1
            elif v149_label == "Complete primary":
                v149_encoded = 2
            elif v149_label == "Incomplete secondary":
                v149_encoded = 3
            elif v149_label == "Complete secondary":
                v149_encoded = 4
            else:
                v149_encoded = 5

            v150_encoded = 0
            if v150_label == "Not related":
                v150_encoded = 0
            elif v150_label == "Adopted/foster child":
                v150_encoded = 1
            elif v150_label == "Other relative":
                v150_encoded = 2
            elif v150_label == "Sister":
                v150_encoded = 3
            elif v150_label == "Mother-in-law":
                v150_encoded = 4
            elif v150_label == "Mother":
                v150_encoded = 5
            elif v150_label == "Grand-daughter":
                v150_encoded = 6
            elif v150_label == "Daughter-in-law":
                v150_encoded = 7
            elif v150_label == "Daughter":
                v150_encoded = 8
            elif v150_label == "Wife":
                v150_encoded = 9
            else:
                v150_encoded = 10

            v504_encoded = 0
            if v504_label == "Staying elsewhere":
                v504_encoded = 0
            elif v504_label == "Living with her":
                v504_encoded = 1
            elif v504_label == "Missing":
                v504_encoded = 8
            else:
                v504_encoded = 9

            v501_never_in_union = 0
            v501_maried = 0
            if v501_one == "Question not applicable":
                v501_never_in_union = 0
                v501_maried = 0
            elif v501_one == "Never in uninon":
                v501_never_in_union = 1
            else:
                v501_maried = 1

            v513_encoded = 0
            if v513_label == "Never married":
                v513_encoded = 0
            elif v513_label == "0-4 years":
                v513_encoded = 1
            elif v513_label == "5-9 years":
                v513_encoded = 2
            elif v513_label == "10-14 years":
                v513_encoded = 3
            elif v513_label == "15-19 years":
                v513_encoded = 4
            elif v513_label == "20-24 years":
                v513_encoded = 5
            else:
                v513_encoded = 6

            v119_encoded = 0
            if v119_label == "Yes":
                v119_encoded = 2
            elif v119_label == "No":
                v119_encoded = 1
            else:
                v119_encoded = 8

            v122_encoded = 0
            if v122_label == "Yes":
                v122_encoded = 2
            elif v122_label == "No":
                v122_encoded = 1
            else:
                v122_encoded = 8

            v121_encoded = 0
            if v121_label == "Yes":
                v121_encoded = 2
            elif v121_label == "No":
                v121_encoded = 1
            else:
                v121_encoded = 8

            v116_encoded = 0
            if v116_label == "Yes":
                v116_encoded = 1
            else:
                v116_encoded = 0

            m15_encoded = 0
            if m15_one == "No":
                m15_encoded = 0
            else:
                m15_encoded = 0

            m3a_encoded = 0
            if m3a_label == "No":
                m3a_encoded = 2
            elif m3a_label == "Yes":
                m3a_encoded = 3
            elif m3a_label == "Missing":
                m3a_encoded = 8
            else:
                m3a_encoded = 9

            m19a_encoded = 0
            if m19a == "Yes":
                m19a_encoded = 1

            v602_have_another = 0
            v602_no_more = 0
            if v602_one == "Have Another":
                v602_have_another = 1
                v602_no_more = 0
            elif v602_one == "No More":
                v602_have_another = 0
                v602_no_more = 1

            v626_not_married_no_sex = 0
            v626_unmet_need_limit = 0
            v626_using_to_limit = 0
            v626_using_to_space = 0
            if v626_one == "Question not applicable":
                v626_not_married_no_sex = 0
                v626_unmet_need_limit = 0
                v626_using_to_limit = 0
                v626_using_to_space = 0
            elif v626_one == "Not married and no sex in last 30 days":
                v626_not_married_no_sex = 1
            elif v626_one == "Unmet need to limit":
                v626_unmet_need_limit = 1
            elif v626_one == "Using to limit":
                v626_using_to_limit = 1
            elif v626_one == "Using to space":
                v626_using_to_space = 1

            v312_pill = 0
            v312_not_using = 0
            if v312_one == "Yes":
                v312_pill = 1
            elif v312_one == "No":
                v312_not_using = 1
            else:
                v312_pill = 0
                v312_not_using = 0

            v364_modern = 0
            v364_non_user_intends = 0
            if v364_one == "Yes":
                v364_modern = 1
            elif v364_one == "Non user but intends to use":
                v364_non_user_intends = 1
            else:
                v364_modern = 0
                v364_non_user_intends = 0

            v362_use_later = 0
            v362_na = 0
            if v362_one == "Yes":
                v362_use_later = 1
            elif v362_one == "No":
                v362_na = 1

            v361_never_used = 0
            v361_used_before_birth = 0
            if v361_one == "Never Used":
                v361_never_used = 1
            elif v361_one == "Used before last birth":
                v361_used_before_birth = 1
            elif v361_one == "Not Applicable":
                v361_never_used = 0
                v361_used_before_birth = 0

            v313_modern = 0
            v313_no_method = 0
            if v313_one == "Modern Method":
                v313_modern = 1
            elif v313_one == "No Method":
                v313_no_method = 1
            elif v313_one == "Not Applicable":
                v313_modern = 0
                v313_no_method = 0

            v327_na = 0
            v327_pharm = 0
            if v327_one == "Yes":
                v327_pharm = 1
            elif v327_one == "No":
                v327_na = 1
            else:
                v327_na = 0
                v327_pharm = 0

            # input_var_df = pd.DataFrame(columns=headers)

            input_values = [v012, v119_encoded, v121_encoded, v122_encoded, v137, v149_encoded, v150_encoded,
                            v201, v202, v203, v219, v337, v504_encoded, v513_encoded, v614, v729_encoded,
                            bord, b2, m3a_encoded, m19, v116_encoded, v312_not_using, v312_pill,
                            v626_not_married_no_sex,
                            v626_unmet_need_limit, v626_using_to_limit, v626_using_to_space, m15_encoded,
                            m19a_encoded, v602_have_another, v602_no_more, v501_maried, v501_never_in_union,
                            v364_non_user_intends, v364_modern, v362_na, v362_use_later, v361_never_used,
                            v361_used_before_birth, v327_na, v327_pharm, v313_modern, v313_no_method]

            print(input_values)

            result = package.predict_infant_mortality(input_values)
            print(result)
            return render_template('index.html', result=result, inf_mort_rate=package.infant_mortality_rate(data, img),
                               inf_mort_rate_yr=package.total_infant_mortality_year(data, img))

        if request.form['btn'] == 'plot':
            region = request.form["region"]
            return redirect(url_for("descriptive_statistics", rgn=region))


@app.route('/<rgn>', methods=['GET'])
def descriptive_statistics(rgn):
    region_value = rgn
    img = BytesIO()
    data = pd.read_csv('dataset/kr-final-cleaned.csv', low_memory=False)
    dataset_rgn = data.loc[data['V024'] == region_value]

    return render_template("plot.html",
                           high_educ_attain_bar_yes=package.highest_educational_attainment_yes(dataset_rgn, img),
                           region_value=rgn,
                           top_10_occu_respo=package.top_10_occupation_respondents(dataset_rgn, img),
                           top_10_occu_part=package.top_10_occupation_partners(dataset_rgn, img),
                           part_high_educ_attain=package.partner_highest_educational_attainment(dataset_rgn, img),
                           per_alloc_bud_earn=package.person_allocating_budget_earnings(dataset_rgn, img),
                           house_amen_rgn=package.household_amenities_region(dataset_rgn, img),
                           resp_pre_car_rgn=package.respondents_prenatal_care_region(dataset_rgn, img),
                           assist_typ_rgn=package.assistance_type_region(dataset_rgn, img),
                           cont_use_int=package.contraceptive_use_intention(dataset_rgn, img)
                           )


# @app.route('/<rgn>')
# def plot(rgn):
#
#     return render_template('plot.html')


if __name__ == '__main__':
    app.use_reloader = False
    app.debug = True
    app.run()
