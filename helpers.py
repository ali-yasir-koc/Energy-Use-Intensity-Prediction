import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


def read_data(train_path, test_path, sub_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sub_path)
    data = pd.concat([train, test], ignore_index = True)
    data.columns = map(str.lower, data.columns)
    return train, test, data, submission


def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def outlier_update(dataframe):
    rep_cols = ["floor_area", "elevation", "precipitation_inches", "snowfall_inches", "snowdepth_inches", "site_eui"]
    for col in rep_cols:
        replace_with_thresholds(dataframe, col)
    return dataframe


def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns, missing_df


def missing_update(dataframe):
    na_cols, missing_df = missing_values_table(dataframe, True)
    drop_cols = missing_df[missing_df["ratio"] > 50].index
    dataframe.drop(columns = drop_cols, inplace = True)
    dataframe['year_built'] = dataframe['year_built'].fillna(dataframe.groupby('state_factor')['year_built'].
                                                             transform('median'))
    dataframe.loc[dataframe["year_built"] == 0, "year_built"] = 1938
    dataframe["year_built"] = dataframe["year_built"].astype("int")
    return dataframe


def get_manual_facility_groups():
    facility_groups = {
        "Living_Space": {
            "2to4_Unit_Building",
            "5plus_Unit_Building",
            "Mixed_Use_Predominantly_Residential",
            "Multifamily_Uncategorized",
            "Mixed_Use_Commercial_and_Residential",
            "Mixed_Use_Predominantly_Commercial",
        },
        "Social_Institutions": {
            "Education_College_or_university",
            "Education_Other_classroom",
            "Education_Preschool_or_daycare",
            "Education_Uncategorized",
            "Health_Care_Inpatient",
            "Health_Care_Outpatient_Clinic",
            "Health_Care_Outpatient_Uncategorized",
            "Health_Care_Uncategorized",
            "Nursing_Home",
            "Religious_worship"
        },
        "Business_Commercial_Venues": {
            "Commercial_Other",
            "Commercial_Unknown",
            "Industrial",
            "Parking_Garage",
            "Food_Sales",
            "Food_Service_Other",
            "Food_Service_Restaurant_or_cafeteria",
            "Food_Service_Uncategorized",
            "Grocery_store_or_food_market",
            "Office_Bank_or_other_financial",
            "Office_Medical_non_diagnostic",
            "Office_Mixed_use",
            "Office_Uncategorized",
            "Retail_Enclosed_mall",
            "Retail_Strip_shopping_mall",
            "Retail_Uncategorized",
            "Retail_Vehicle_dealership_showroom",
            "Laboratory",
            "Data_Center",
            "Lodging_Dormitory_or_fraternity_sorority",
            "Lodging_Hotel",
            "Lodging_Other",
            "Lodging_Uncategorized",
        },
        "Public": {
            "Public_Assembly_Drama_theater",
            "Public_Assembly_Entertainment_culture",
            "Public_Assembly_Library",
            "Public_Assembly_Movie_Theater",
            "Public_Assembly_Other",
            "Public_Assembly_Recreation",
            "Public_Assembly_Social_meeting",
            "Public_Assembly_Stadium",
            "Public_Assembly_Uncategorized",
            "Public_Safety_Courthouse",
            "Public_Safety_Fire_or_police_station",
            "Public_Safety_Penitentiary",
            "Public_Safety_Uncategorized",
        },
        "Warehouse_Service": {
            "Warehouse_Distribution_or_Shipping_center",
            "Warehouse_Nonrefrigerated",
            "Warehouse_Refrigerated",
            "Warehouse_Selfstorage",
            "Warehouse_Uncategorized",
            "Service_Drycleaning_or_Laundry",
            "Service_Uncategorized",
            "Service_Vehicle_service_repair_shop",
        },
    }

    return facility_groups


def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first, dtype = "int")
    return dataframe


def generate_features(dataframe):
    dataframe["build_age"] = 2023 - dataframe["year_built"]
    dataframe.drop(columns = "year_built", inplace = True)

    facility_groups = get_manual_facility_groups()
    dataframe['category'] = dataframe['facility_type'].apply(lambda x:
                                                             next((category for category, values in
                                                                   facility_groups.items()
                                                                   if x in values), None))
    dataframe.drop(columns = ["facility_type", "building_class"], inplace = True)

    ## The energy_star_rating column is organized according to the category column.
    dataframe['energy_star_rating'] = dataframe['energy_star_rating']. \
        fillna(dataframe.groupby(["state_factor", "category"])['energy_star_rating'].
               transform('median'))
    dataframe['energy_star_rating'] = dataframe['energy_star_rating'].fillna(dataframe["energy_star_rating"].median())

    temp_min_max = dataframe.groupby("state_factor")["january_min_temp"].min().reset_index()
    for i in dataframe.columns[6:41]:
        if "_min_" in i:
            temp_min_max[i + "_for_state"] = dataframe.groupby("state_factor")[i].min().values
        elif "_avg_" in i:
            temp_min_max[i + "_for_state"] = dataframe.groupby("state_factor")[i].mean().values
        elif "_max_" in i:
            temp_min_max[i + "_for_state"] = dataframe.groupby("state_factor")[i].max().values

    temp_min_max["min_temp_for_state"] = temp_min_max.iloc[:, 1:].min(axis = 1)
    temp_min_max["max_temp_for_state"] = temp_min_max.iloc[:, 1:].max(axis = 1)
    dataframe = pd.merge(dataframe,
                         temp_min_max[["state_factor", "min_temp_for_state", "max_temp_for_state"]],
                         how = "left",
                         on = "state_factor")

    dataframe["floor_prod_elevation"] = dataframe["floor_area"] * dataframe["elevation"]

    ohe_cols = ["state_factor", "category"]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
    dataframe.drop(columns = ["id", "year_factor"], inplace = True)

    target = "site_eui"
    standard_cols = [col for col in dataframe.columns if col != target]
    ss = StandardScaler()
    for col in standard_cols:
        dataframe[col] = ss.fit_transform(dataframe[[col]])

    dataframe[target] = np.log1p(dataframe[target])
    return dataframe


def split_data(dataframe):
    y = dataframe[dataframe["site_eui"].notna()]["site_eui"]
    X = dataframe[dataframe["site_eui"].notna()].drop("site_eui", axis = 1)
    return X, y


def create_model(X, y):
    lgbm_model = LGBMRegressor(colsample_bytree=0.5,
                               learning_rate = 0.01,
                               n_estimators = 1000,
                               metric = "rmse",
                               random_state = 17).fit(X, y)
    return lgbm_model


def prediction(model, dataframe, submission_dataframe):
    test_final = dataframe[dataframe["site_eui"].isna()].reset_index(drop = True)
    test_final.drop(columns = "site_eui", inplace = True)
    predict = model.predict(test_final)
    predict = np.expm1(predict)
    submission_dataframe["site_eui"] = predict
    submission_dataframe.to_csv("datasets/wids_submission.csv", index = False)
    print(" --- Results are saved ---")
    return submission_dataframe
