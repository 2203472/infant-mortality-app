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
import xgboost


def predict_infant_mortality(input_values):
    print('Nakapasok sa model')
    # Use pickle to load in the pre-trained model.
    with open(f'model/infant-mortality-xgboost-model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Use pickle to load the PCA model.
    with open(f'dataset/kr-final-pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    final_values = list()

    # Standardization
    means = [30.020727591613298, 1.7467779224575484, 1.57891100422524, 1.2668810289389068, 1.8438255693444234,
             3.3114719247428983, 8.513459647630942, 3.467540060056868, 1.5715766256543806, 1.4499481810209667,
             3.3615370306396324, 55.56881294677261, 1.3586139087454492, 2.40910419600861, 3.235603624671149,
             3.156679333528208, 3.1351013791820574, 13.03178230714039, 2.366001434987112, 5.059594695863676,
             0.4089713268316016, 0.4738380590470623, 0.18785044245435945, 0.025457734314793656, 0.13969865270654513,
             0.34192554011320453, 0.1842364008397332, 0.46977226223060775, 0.25946692886184264, 0.31078100502245487,
             0.5685206345831894, 0.740320480454944, 0.013977837421275013, 0.26015784858228586, 0.3610321277670006,
             0.5264011054715527, 0.2117668943158566, 0.2154872312720895, 0.13504823151125403, 0.6423693231644123,
             0.05628338338072334, 0.3610321277670006, 0.4738380590470623]

    stdevs = [6.731180559267618, 0.43614425709578575, 0.49486933862999427, 0.4498416713335693, 0.8874649386318114,
              1.4422374397381885, 1.4187624409665531, 2.331018239306285, 1.3171426240046313, 1.2605913493538428,
              2.1407942370537736, 42.52028937116891, 1.8383916622527698, 1.3393098053782857, 1.2994933318703923,
              1.745486709953654, 2.279529958566367, 7.389835777865726, 0.5557914404915119, 3.2411105714294184,
              0.4916504897732531, 0.49932171822601823, 0.39059788530730033, 0.15751284831117976, 0.3466787171792647,
              0.4743610910840786, 0.3876820132817717, 0.4990920789023669, 0.4383479757262574, 0.46281947247784305,
              0.49528925033981336, 0.4384645658717301, 0.11740027148519873, 0.4387264037619226, 0.4803062157757754,
              0.4993091294402239, 0.4085659220427584, 0.4111653887708846, 0.34177962322910616, 0.47930896175349064,
              0.2304712035323833, 0.4803062157757754, 0.49932171822601823]

    for i in range(len(input_values)):
        input_values[i] = (input_values[i] - means[i]) / stdevs[i]
    final_values.append(input_values)
    standardized_values = pd.DataFrame(final_values)

    reduced_values = pca.transform(standardized_values)
    input_variables = pd.DataFrame(data=reduced_values, columns=['PC 1', 'PC 2'])
    # Normalization of user input
    # dataset = pd.read_csv('dataset/kr-final-transformed.csv', low_memory=False)
    # scaled_df = input_var_df.copy()
    # for col in scaled_df.columns:
    #     scaled_df[col] = scaled_df[col].apply(
    #         lambda x: (x - dataset[col].min()) / (dataset[col].max() - dataset[col].min()))
    # input_variables = scaled_df.tail(1)

    # PCA
    prediction = model.predict(input_variables)[0]
    # prediction = model.predict(input_variables, predict_disable_shape_check=True)[0]
    if prediction == 0:
        printpredict = "Positive Infant Mortality"
    else:
        printpredict = "Negative Infant Mortality"

    return printpredict

# Highest Education Attainment for each Mortality class

def highest_educational_attainment_yes(dataset_rgn, img):
    # Bar Plot Highest Education Attainment for yes class
    yes_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'] == 'Yes']

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


def highest_educational_attainment_no(dataset_rgn, img):
    # Bar Plot Highest Education Attainment for no class
    no_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'] == 'No']
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


def income(dataset_rgn, img):
    # Income
    no_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'] == 'No']
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


def partner_highest_educational_attainment(dataset_rgn, img):
    # Husband/Partner's Highest Educational Attainment
    no_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'] == 'No']
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


def person_allocating_budget_earnings(dataset_rgn, img):
    # Person in-charge of allocating budget of respondent's earnings
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896']

    no_b5_class = dataset_rgn.loc[dataset_rgn['B5CLASS'] == 'No']
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
