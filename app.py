import math

import pickle

import flask
import pandas as pd
from scipy.stats import stats
from sklearn import preprocessing
import seaborn as sns

from flask import request, url_for, redirect, render_template
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# # Use pickle to load in the pre-trained model.
# with open(f'model/povery-prediction-lightgbm.pkl', 'rb') as f:
#     model = pickle.load(f)

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')

dataset = pd.read_csv('dataset/kr-final-cleaned.csv', low_memory=False)
yes_b5_class = dataset.loc[dataset['B5CLASS'].str.contains("Yes")]
no_b5_class = dataset.loc[dataset['B5CLASS'].str.contains("No")]
img = BytesIO()


# Highest Education Attainment for each Mortality class

def highest_educational_attainment():
    # Bar Plot Highest Education Attainment for yes class
    no_class_education_count = no_b5_class['V149'].value_counts()

    plt.figure(figsize=(15, 7))
    ax1 = sns.barplot(x=no_class_education_count.index, y=no_class_education_count.values, palette="pastel",
                      edgecolor=".6", hue=no_class_education_count.index, dodge=False)
    plt.ylabel('Total Count')
    plt.title('Highest Education Attainment for Families having Positive Infant Mortality', fontsize=20)

    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    high_educ_attain_bar = base64.b64encode(img.getvalue()).decode('utf8')

    return high_educ_attain_bar


def income():
    # Income
    no_class_occupation_count = no_b5_class['V717'].value_counts()
    no_class_occupation_count = no_class_occupation_count[:10, ]

    plt.figure(figsize=(15, 7))
    ax1 = sns.barplot(x=no_class_occupation_count.index, y=no_class_occupation_count.values, palette="pastel",
                      edgecolor=".6", hue=no_class_occupation_count.index, dodge=False)
    plt.ylabel('Total Count')
    plt.title("Top 10 Occupations of Respondent's Having Positive Infant Mortality", fontsize=20)

    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    inc = base64.b64encode(img.getvalue()).decode('utf8')

    return inc


def partner_highest_educational_attainment():
    # Husband/Partner's Highest Educational Attainment
    no_class_partner_education_count = no_b5_class['V729'].value_counts()

    plt.figure(figsize=(15, 7))
    ax1 = sns.barplot(x=no_class_partner_education_count.index, y=no_class_partner_education_count.values,
                      palette="pastel",
                      edgecolor=".6", hue=no_class_partner_education_count.index, dodge=False)
    plt.ylabel('Total Count')
    plt.title("Husband/Partner's Highest Education Attainment Having Positive Infant Mortality", fontsize=20)

    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    part_high_educ_attain = base64.b64encode(img.getvalue()).decode('utf8')

    return part_high_educ_attain


def person_allocating_budget_earnings():
    # Person in-charge of allocating budget of respondent's earnings
    explode = (0, 0, 1, 1, 1)
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    no_class_person_allocate_budget_count = no_b5_class['V739'].value_counts()
    no_class_person_allocate_budget_list = list(no_class_person_allocate_budget_count.keys())

    person_in_charge = []

    for key in no_class_person_allocate_budget_count.keys():
        person_in_charge.append(no_class_person_allocate_budget_count[key])

    plt.figure(figsize=(10, 10))

    plt.pie(person_in_charge, labels=no_class_person_allocate_budget_list,
            shadow=False, startangle=180, colors=colors, explode=explode,
            autopct='%1.1f%%')

    plt.title('Person in-charge of Allocating Budget of Earnings',
              fontname="Century Gothic",
              size=18)
    plt.tight_layout()

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    per_alloc_bud_earn = base64.b64encode(img.getvalue()).decode('utf8')

    return per_alloc_bud_earn


def infant_mortality_rate():
    # Line Graph of Infant Mortality
    dataset['B5CLASS'].replace(['Yes', 'No'], [1, 0], inplace=True)

    one = (288 / 7165) * 100
    two = (207 / 6894) * 100
    three = (159 / 5945) * 100
    four = (167 / 6565) * 100
    five = (219 / 10303) * 100

    values = [one, two, three, four, five]
    years = ["1998", "2003", "2008", "2013", "2017"]

    plt.figure(figsize=(15, 7))
    plt.plot(years, values)
    plt.xlabel('Year')
    plt.ylabel('Total Positive Infant Mortality')
    plt.title("Total Infant Mortality for Each Year", fontsize=20)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    inf_mort_rate = base64.b64encode(img.getvalue()).decode('utf8')

    return inf_mort_rate


def total_infant_mortality_year():
    # Get different years with
    # yes value via value counts
    no_class_year_count = no_b5_class['V007'].value_counts()

    plt.figure(figsize=(15, 7))
    ax1 = sns.barplot(x=no_class_year_count.index, y=no_class_year_count.values, palette="pastel",
                      edgecolor=".6", hue=no_class_year_count.index, dodge=False)
    plt.xlabel('Year')
    plt.ylabel('Total Positive Infant Mortality')
    plt.title("Total Infant Mortality for Each Year", fontsize=20)

    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    inf_mort_rate_yr = base64.b64encode(img.getvalue()).decode('utf8')

    return inf_mort_rate_yr


def household_amenities_region():
    # Household Amenities per Region
    amenities = dataset[['V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125']]
    variables = {'V119': "Electricity", 'V120': "Radio", 'V121': "Television", 'V122': "Refrigerator",
                 'V123': "Bicycle", 'V124': "Motorcycle/Scooter", 'V125': "Car/Truck"}

    amenities = amenities.rename(columns=variables).apply(pd.Series.value_counts)

    amenities_reshaped = pd.melt(amenities, var_name="Amenities", value_name="Quantity", ignore_index=False)
    amenities_reshaped['Ownership'] = amenities_reshaped.index
    amenities_reshaped = amenities_reshaped[amenities_reshaped['Ownership'] != 'Missing']

    colors = ['#DF9D9E', '#BF899C']
    plt.figure(figsize=(15, 7))

    plot = sns.catplot(x="Quantity", y="Amenities", hue="Ownership", data=amenities_reshaped,
                       kind='bar', aspect=2.5, dodge=False, palette=colors)
    plot.fig.suptitle("Household Amenities per Region", font="Century Gothic", fontsize=20)
    plt.tight_layout()

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    house_amen_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    return house_amen_rgn


def respondents_prenatal_care_region():
    # Number of Respondents Receiving Prenatal Care per Region
    prenatal = dataset[['M2A', 'M2K', 'M2N']]
    variables = {'M2A': "Doctor", 'M2K': "Other", 'M2N': "No One"}

    prenatal = prenatal.rename(columns=variables).apply(pd.Series.value_counts)

    prenatal_reshaped = pd.melt(prenatal, var_name="Prenatal Care Type", value_name="Quantity", ignore_index=False)
    prenatal_reshaped['Taken Care Of'] = prenatal_reshaped.index
    prenatal_reshaped = prenatal_reshaped[(prenatal_reshaped['Taken Care Of'] != 'Missing')]

    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    plt.figure(figsize=(10, 7))
    sns.catplot(x="Quantity", y="Prenatal Care Type", hue="Taken Care Of", data=prenatal_reshaped,
                kind='bar', aspect=2, dodge=False, palette=colors, legend_out=False)
    plt.title("Number of Respondents Receiving Prenatal Care per Region", font="Century Gothic", fontsize=18)
    plt.tight_layout()

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    resp_pre_car_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    return resp_pre_car_rgn


def assistance_type_region():
    # Assistance Type per Region
    assistance = dataset[['M3A', 'M3H', 'M3K', 'M3N']]
    variables = {'M3A': "Doctor", 'M3H': "Barangay Health Worker", 'M3K': "Other", 'M3N': "No One"}

    assistance = assistance.rename(columns=variables).apply(pd.Series.value_counts)

    assistance_reshaped = pd.melt(assistance, var_name="Assistance Type", value_name="Quantity", ignore_index=False)
    assistance_reshaped['Assisted'] = assistance_reshaped.index
    assistance_reshaped = assistance_reshaped[(assistance_reshaped['Assisted'] != 'Missing')]

    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    plt.figure(figsize=(15, 7))
    sns.catplot(x="Quantity", y="Assistance Type", hue="Assisted", data=assistance_reshaped,
                kind='bar', aspect=2, dodge=False, palette=colors, legend_out=False)
    plt.title("Assistance Type per Region", font="Century Gothic", fontsize=20)
    plt.tight_layout()

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    assist_typ_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    return assist_typ_rgn


def contraceptive_use_intention():
    # Contraceptive Use and Intention
    explode = (0.1, 0.1, 0.1, 0.2)
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    no_class_contraceptive_use_intention_count = dataset['V364'].value_counts()
    no_class_contraceptive_use_intention_list = list(no_class_contraceptive_use_intention_count.keys())

    contraceptive_use_intention = []

    for key in no_class_contraceptive_use_intention_count.keys():
        contraceptive_use_intention.append(no_class_contraceptive_use_intention_count[key])

    plt.figure(figsize=(8, 8))
    plt.pie(contraceptive_use_intention, labels=no_class_contraceptive_use_intention_list,
            shadow=False, startangle=180, colors=colors, explode=explode, autopct='%1.1f%%')
    plt.title('Contraceptive Use and Intention',
              fontname="Century Gothic",
              size=18)
    plt.tight_layout()

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    cont_use_int = base64.b64encode(img.getvalue()).decode('utf8')

    return cont_use_int


@app.route('/', methods=['GET', 'POST'])
def index():

    if flask.request.form['btn'] == 'predict':

        return render_template()
    # with open(f'datasets/raw-features.pkl', 'rb') as f:
    #     df = pickle.load(f)

    # if flask.request.method == 'GET':
    #     return flask.render_template('main.html')

    # if flask.request.method == 'POST':
    #     if request.form['btn'] == 'predict':
    #         return render_template('main.html')

    # if request.form['btn'] == 'Plot':
    #     region = request.form["region"]
    #     return redirect(url_for("plot", rgn=region))

    return render_template("sample.html", high_educ_attain_bar=highest_educational_attainment(),
                           inc=income(),
                           part_high_educ_attain=partner_highest_educational_attainment(),
                           per_alloc_bud_earn=person_allocating_budget_earnings(),
                           inf_mort_rate=infant_mortality_rate(),
                           inf_mort_rate_yr=total_infant_mortality_year(),
                           house_amen_rgn=household_amenities_region(),
                           resp_pre_car_rgn=respondents_prenatal_care_region(),
                           assist_typ_rgn=assistance_type_region(),
                           cont_use_int=contraceptive_use_intention()
                           )


# @app.route('/<rgn>')
# def plot(rgn):
#
#     return render_template('plot.html')


if __name__ == '__main__':
    app.run()
