########################## Library Importing and Settings ###########################
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


########################## Data Loading From Local  ###########################
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
all_data = pd.concat([train, test], ignore_index = True)
data = all_data.copy()
data.head()
data.columns = map(str.lower, data.columns)


########################## Summary of  The Data  ###########################
def check_df(dataframe):
    if isinstance(dataframe, pd.DataFrame):
        print("########## shape #########\n", dataframe.shape)
        print("########## types #########\n", dataframe.dtypes)
        print("########## head #########\n", dataframe.head())
        print("########## tail #########\n", dataframe.tail())
        print("########## NA #########\n", dataframe.isna().sum())
        print("########## describe #########\n", dataframe.describe().T)
        print("########## nunique #########\n", dataframe.nunique())


check_df(data)


def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


columns_info(data)


########################## Grab to Columns ###########################
def grab_col_names(dataframe, cat_th = 8, car_th = 62):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return list(cat_cols), list(num_cols), list(cat_but_car)


cat_cols, num_cols, cat_but_car = grab_col_names(data)


########################## Variables Analysis ###########################
def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)


for col in cat_cols:
    cat_summary(data, col, True)

def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.histplot(dataframe[numerical_col])
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)


for col in num_cols:
    num_summary(data, col, plot = True)


########################## Target Analysis ###########################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")


for col in cat_cols:
    target_summary_with_cat(data, "site_eui", col)


########################## Outlier Analysis ###########################
def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(data, col))

have_outlier_cols = [col for col in num_cols if check_outlier(data, col)]
data[have_outlier_cols].head()
rep_cols = ["floor_area", "elevation", "precipitation_inches", "snowfall_inches",  "snowdepth_inches", "site_eui"]

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in rep_cols:
    replace_with_thresholds(data, col)
# Outliers in rep_cols columns replaced with threshold values.


########################## Missing Value Analysis ###########################
def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns, missing_df


na_cols, missing_df = missing_values_table(data, True)
drop_cols = missing_df[missing_df["ratio"] > 50].index
data.drop(columns=drop_cols, inplace=True)

data['year_built'] = data['year_built'].fillna(data.groupby('state_factor')['year_built'].transform('median'))
data.loc[data["year_built"] == 0, "year_built"] = 1938
data["year_built"] = data["year_built"].astype("int")


########################## Feature Engineering ###########################
""" 1- Age Build """
data["build_age"] = 2023 - data["year_built"]
data.drop(columns ="year_built", inplace = True)

""" 2- Category"""
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


facility_groups = get_manual_facility_groups()
data['category'] = data['facility_type'].apply(lambda x:
                                               next((category for category, values in facility_groups.items()
                                                     if x in values), None))
data.drop(columns = ["facility_type", "building_class"], inplace = True)
data.head()

## The energy_star_rating column is organized according to the category column.
data['energy_star_rating'] = data['energy_star_rating']. \
    fillna(data.groupby(["state_factor", "category"])['energy_star_rating'].
           transform('median'))
data['energy_star_rating'] = data['energy_star_rating'].fillna(data["energy_star_rating"].median())

""" 3- Annual minimum and maximum temperatures """
temp_min_max = data.groupby("state_factor")["january_min_temp"].min().reset_index()
for i in data.columns[6:41]:
    if "_min_" in i:
        temp_min_max[i+"_for_state"] = data.groupby("state_factor")[i].min().values
    elif "_avg_" in i:
        temp_min_max[i+"_for_state"] = data.groupby("state_factor")[i].mean().values
    elif "_max_" in i:
        temp_min_max[i + "_for_state"] = data.groupby("state_factor")[i].max().values

temp_min_max["min_temp_for_state"] = temp_min_max.iloc[:, 1:].min(axis=1)
temp_min_max["max_temp_for_state"] = temp_min_max.iloc[:, 1:].max(axis=1)

data = pd.merge(data,
                temp_min_max[["state_factor", "min_temp_for_state", "max_temp_for_state"]],
                how ="left",
                on= "state_factor")

""" 4- Floor Area Product Elevation"""
data["floor_prod_elevation"] = data["floor_area"] * data["elevation"]

""" 5- Get Dummies Columns """
def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first, dtype = "int")
    return dataframe


ohe_cols = ["state_factor", "category"]
data[ohe_cols].head()

new_df = one_hot_encoder(data, ohe_cols)
new_df.head()
new_df.drop(columns = ["id", "year_factor"], inplace=True)

""" 6- Standardization """
target = "site_eui"
standard_cols = [col for col in new_df.columns if col != target]
ss = StandardScaler()
for col in standard_cols:
    new_df[col] = ss.fit_transform(new_df[[col]])

new_df[target] = np.log1p(new_df[target])
new_df.head()
print(new_df.shape)


########################## Model Selection ###########################
y = new_df[new_df["site_eui"].notna()]["site_eui"]
X = new_df[new_df["site_eui"].notna()].drop("site_eui", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor(random_state = 17)),
          ('RF', RandomForestRegressor(random_state = 17)),
          ('GBM', GradientBoostingRegressor(random_state = 17)),
          ("XGBoost", XGBRegressor(eval_metric= "rmse", objective = 'reg:squarederror', random_state=17)),
          ("LightGBM", LGBMRegressor(metric= "rmse", objective='regression', random_state = 17))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv = 5, scoring = "neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


########################## Final Model ###########################
### LightGBM
lgbm_model = LGBMRegressor(metric = "rmse", random_state = 17)
lgbm_params = {"learning_rate": [0.01, 0.1, 0.2],
               "n_estimators": [500, 1000, 2000, 2500],
               "colsample_bytree": [0.5, 0.7, 1]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv = 5,
                            n_jobs = -1,
                            verbose = True).fit(X, y)
best = lgbm_gs_best.best_params_
print(lgbm_gs_best.best_params_)

final_model = lgbm_model.set_params(**best).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv = 5, scoring = "neg_mean_squared_error")))
print(rmse)


########################## Features Importance ###########################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(final_model, X)


########################## Prediction ###########################
submission = pd.read_csv("datasets/sample_solution.csv")
submission.head()
test_final = new_df[new_df["site_eui"].isna()].reset_index(drop = True)
test_final.drop(columns="site_eui", inplace = True)
predict = final_model.predict(test_final)
predict = np.expm1(predict)
submission["site_eui"] = predict
submission.to_csv("datasets/wids_submission.csv", index = False)


########################## Alternative Models ###########################
## GBM
gbm_model = GradientBoostingRegressor(random_state=17)
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
best_gmb = {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 500, "subsample": 0.7}
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

## RF
rf_model = RandomForestRegressor(random_state=17)
rf_params = {"max_depth": [8, 10, None],
             "max_features": [0.5, 0.7, "auto"],
             "min_samples_split": [3, 5, 8, 12],
             "n_estimators": [100, 200, 500]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv = 5, scoring = "neg_mean_squared_error")))

