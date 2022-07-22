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

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')

data = pd.read_csv('dataset/kr-final-cleaned.csv', low_memory=False)
img = BytesIO()


def predict_infant_mortality(input_var_df):
    # Use pickle to load in the pre-trained model.
    with open(f'model/infant-mortality-xgboost-model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Normalization of user input
    dataset = pd.read_csv('dataset/kr-final-transformed.csv', low_memory=False)
    scaled_df = input_var_df.copy()
    for col in scaled_df.columns:
        scaled_df[col] = scaled_df[col].apply(
            lambda x: (x - dataset[col].min()) / (dataset[col].max() - dataset[col].min()))
    input_variables = scaled_df.tail(1)

    # PCA

    prediction = model.predict(input_variables, predict_disable_shape_check=True)[0]
    return prediction


# Highest Education Attainment for each Mortality class

def highest_educational_attainment_yes(yes_b5_class, img):
    # Bar Plot Highest Education Attainment for yes class
    no_class_education_count = yes_b5_class['V149'].value_counts()

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


def highest_educational_attainment_no(no_b5_class, img):
    # Bar Plot Highest Education Attainment for no class
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


def income(no_b5_class, img):
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


def partner_highest_educational_attainment(no_b5_class, img):
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


def person_allocating_budget_earnings(no_b5_class, img):
    # Person in-charge of allocating budget of respondent's earnings
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    no_class_person_allocate_budget_count = no_b5_class['V739'].value_counts()
    no_class_person_allocate_budget_list = list(no_class_person_allocate_budget_count.keys())

    num_index = len(no_class_person_allocate_budget_count.index)
    temp = []
    for i in range(0, num_index):
        value = 0.1
        temp.append(value)

    explode = tuple(temp)

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


def infant_mortality_rate(dataset, img):
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


def total_infant_mortality_year(no_b5_class, img):
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


def household_amenities_region(dataset, img):
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


def respondents_prenatal_care_region(dataset, img):
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


def assistance_type_region(dataset, img):
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


def contraceptive_use_intention(dataset, img):
    # Contraceptive Use and Intention
    # explode = (0.1, 0.1, 0.1, 0.2)
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    no_class_contraceptive_use_intention_count = dataset['V364'].value_counts()
    no_class_contraceptive_use_intention_list = list(no_class_contraceptive_use_intention_count.keys())
    num_index = len(no_class_contraceptive_use_intention_count.index)
    temp = []
    for i in range(0, num_index):
        value = 0.1
        temp.append(value)

    explode = tuple(temp)

    contraceptive_use_intention_list = []

    for key in no_class_contraceptive_use_intention_count.keys():
        contraceptive_use_intention_list.append(no_class_contraceptive_use_intention_count[key])

    plt.figure(figsize=(8, 8))
    plt.pie(contraceptive_use_intention_list, labels=no_class_contraceptive_use_intention_list,
            shadow=False, startangle=180, colors=colors, autopct='%1.1f%%', explode=explode)
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
    if flask.request.method == 'GET':
        return render_template('index.html', inf_mort_rate=infant_mortality_rate(data, img),
                               inf_mort_rate_yr=total_infant_mortality_year(data, img))

    if flask.request.method == 'POST':
        if request.form['btn'] == 'predict':
            V012 = (flask.request.form['V012'])
            V119 = (flask.request.form['V119'])
            V121 = (flask.request.form['V121'])
            V122 = (flask.request.form['V122'])
            V137 = (flask.request.form['V137'])
            V149 = (flask.request.form['V149'])
            V150 = (flask.request.form['V150'])
            V201 = (flask.request.form['V201'])
            V202 = (flask.request.form['V202'])
            V203 = (flask.request.form['V203'])
            V219 = (flask.request.form['V219'])
            V337 = (flask.request.form['V337'])
            V504 = (flask.request.form['V504'])
            V513 = (flask.request.form['V513'])
            V614 = (flask.request.form['V614'])
            V729 = (flask.request.form['V729'])
            BORD = (flask.request.form['BORD'])
            B2 = (flask.request.form['B2'])
            M3A = (flask.request.form['M3A'])
            M19 = (flask.request.form['M19'])
            V116_Flush_toilet_to_septic_tank = (flask.request.form['V116_Flush_toilet_to_septic_tank'])
            V312_Not_using = (flask.request.form['V312_Not_using'])
            V312_Pill = (flask.request.form['V312_Pill'])
            V326_NA = (flask.request.form['V326_NA'])
            V326_Pharmacy = (flask.request.form['V326_Pharmacy'])
            V363_NA = (flask.request.form['V363_NA'])
            V363_Pill = (flask.request.form['V363_Pill'])
            V376_NA = (flask.request.form['V376_NA'])
            V626_Not_married_and_no_sex_in_last_30_days = (
                flask.request.form['V626_Not_married_and_no_sex_in_last_30_days'])
            V626_Unmet_need_to_limit = (flask.request.form['V626_Unmet_need_to_limit'])
            V626_Using_to_limit = (flask.request.form['V626_Using_to_limit'])
            V626_Using_to_space = (flask.request.form['V626_Using_to_space'])
            V705_NA = (flask.request.form['V705_NA'])
            M15_Respondent_home = (flask.request.form['M15_Respondent_home'])
            M19A_Not_weighed = (flask.request.form['M19A_Not_weighed'])
            V616_NA = (flask.request.form['V616_NA'])
            V616_Years = (flask.request.form['V616_Years'])
            V602_Have_another = (flask.request.form['V602_Have_another'])
            V602_No_more = (flask.request.form['V602_No_more'])
            V501_Married = (flask.request.form['V501_Married'])
            V501_Never_in_union = (flask.request.form['V501_Never_in_union'])
            V364_Non_user_intend_to = (flask.request.form['V364_Non_user_intend_to'])
            V364_Using_modern_method = (flask.request.form['V364_Using_modern_method'])
            V362_NA = (flask.request.form['V362_NA'])
            V362_Use_later = (flask.request.form['V362_Use_later'])
            V361_Never_used = (flask.request.form['V361_Never_used'])
            V361_Used_before_last_birth = (flask.request.form['V361_Used_before_last_birth'])
            V327_NA = (flask.request.form['V327_NA'])
            V327_Pharmacy = (flask.request.form['V327_Pharmacy'])
            V313_Modern_method = (flask.request.form['V313_Modern_method'])
            V313_No_method = (flask.request.form['V313_No_method'])
            V604_NA = (flask.request.form['V604_NA'])

            headers = ['V012', 'V119', 'V121', 'V122', 'V137', 'V149', 'V150', 'V201', 'V202',
                       'V203', 'V219', 'V337', 'V504', 'V513', 'V614', 'V729', 'BORD', 'B2',
                       'M3A', 'M19', 'V116_Flush toilet to septic tank', 'V312_Not using',
                       'V312_Pill', 'V326_NA', 'V326_Pharmacy', 'V363_NA', 'V363_Pill',
                       'V376_NA', 'V626_Not married and no sex in last 30 days',
                       'V626_Unmet need to limit', 'V626_Using to limit',
                       'V626_Using to space', 'V705_NA', "M15_Respondent's home",
                       'M19A_Not weighed', 'V616_NA', 'V616_Year/s', 'V602_Have another',
                       'V602_No more', 'V501_Married', 'V501_Never in union',
                       'V364_Non-user intend to', 'V364_Using modern method', 'V362_NA',
                       'V362_Use later', 'V361_Never used', 'V361_Used before last birth',
                       'V327_NA', 'V327_Pharmacy', 'V313_Modern method', 'V313_No method',
                       'V604_NA']

            input_var_df = pd.DataFrame(columns=headers)

            input_var_df.loc[len(input_var_df)] = [V012, V119, V121, V122, V137, V149, V150, V201, V202,
                                                   V203, V219, V337, V504, V513, V614, V729, BORD, B2,
                                                   M3A, M19, V116_Flush_toilet_to_septic_tank, V312_Not_using,
                                                   V312_Pill, V326_NA, V326_Pharmacy, V363_NA, V363_Pill,
                                                   V376_NA, V626_Not_married_and_no_sex_in_last_30_days,
                                                   V626_Unmet_need_to_limit, V626_Using_to_limit,
                                                   V626_Using_to_space, V705_NA, M15_Respondent_home,
                                                   M19A_Not_weighed, V616_NA, V616_Years, V602_Have_another,
                                                   V602_No_more, V501_Married, V501_Never_in_union,
                                                   V364_Non_user_intend_to, V364_Using_modern_method, V362_NA,
                                                   V362_Use_later, V361_Never_used, V361_Used_before_last_birth,
                                                   V327_NA, V327_Pharmacy, V313_Modern_method, V313_No_method,
                                                   V604_NA]

            result = predict_infant_mortality(input_var_df)
            return render_template('index.html', result=result)

    if request.form['btn'] == 'plot':
        region = request.form["region"]
        return redirect(url_for("descriptive_statistics", rgn=region))


@app.route('/<rgn>')
def descriptive_statistics(rgn):
    region_value = rgn
    dataset_rgn = data.copy()
    dataset_rgn = dataset_rgn.loc[dataset_rgn['V024'] == region_value]
    yes_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'].str.contains('Yes')]
    no_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'].str.contains('No')]

    return render_template("index.html",
                           high_educ_attain_bar_yes=highest_educational_attainment_yes(yes_b5_class, img),
                           high_educ_attain_bar_no=highest_educational_attainment_no(no_b5_class, img),
                           inc=income(no_b5_class, img),
                           part_high_educ_attain=partner_highest_educational_attainment(no_b5_class, img),
                           per_alloc_bud_earn=person_allocating_budget_earnings(no_b5_class, img),
                           house_amen_rgn=household_amenities_region(dataset_rgn, img),
                           resp_pre_car_rgn=respondents_prenatal_care_region(dataset_rgn, img),
                           assist_typ_rgn=assistance_type_region(dataset_rgn, img),
                           cont_use_int=contraceptive_use_intention(dataset_rgn, img)
    )



# @app.route('/<rgn>')
# def plot(rgn):
#
#     return render_template('plot.html')


if __name__ == '__main__':
    app.run()
