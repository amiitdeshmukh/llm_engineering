import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from pricer.evaluator import evaluate
from pricer.items import Item

LITE_MODE = False

username = "ed-donner"
dataset =f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

# --------------------------------------------------------------

# def random_pricer(item):
#     return random.randrange(1,1000)

# random.seed(42)

# evaluate(random_pricer, test)

# --------------------------------------------------------------

# training_prices = [item.price for item in train]
# training_avg = sum(training_prices)/len(training_prices)

# print(training_avg)

# def constant_pricer(item):
#     return training_avg

# evaluate(constant_pricer, test)

# --------------------------------------------------------------


# def get_features(item):
#     return {
#         "weight": item.weight,
#         "weight_unknown": 1 if item.weight==0 else 0,
#         "text_length": len(item.summary)
#     }

# def list_of_dataframe(items):
#     features = [get_features(item) for item in items]
#     df= pd.DataFrame(features)
#     df['price'] = [item.price for item in items]
#     return df

# train_df = list_of_dataframe(train)
# test_df = list_of_dataframe(test)

# np.random.seed(42)

# feature_column = ['weight', 'weight_unknown', 'text_length']

# X_train = train_df[feature_column]
# Y_train = train_df['price']

# X_test = test_df[feature_column]
# Y_test = test_df['price']

# model = LinearRegression()
# model.fit(X_train, Y_train)

# for feature, coef in zip(feature_column, model.coef_):
#     print(f"{feature}: {coef}")

# print(f"Intercept: {model.intercept_}")

# #predict
# y_pred = model.predict(X_test)
# mse = mean_squared_error(Y_test, y_pred)
# r2 = r2_score(Y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared Score: {r2}")

# def linear_regression_pricer(item):
#     features = get_features(item)
#     feature_df= pd.DataFrame([features])
#     return model.predict(feature_df)[0]

# evaluate(linear_regression_pricer, test)


# --------------------------------------------------------------

prices = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]

# np.random.seed(42)
vectorizor = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizor.fit_transform(documents)

# selected_words = vectorizor.get_feature_names_out()
# print(f"Number of selected words: {len(selected_words)}")
# print("Selected words:", selected_words[1000:1020])

# regressor = LinearRegression()
# regressor.fit(X, prices)

# def natural_language_linear_regression_pricer(item):
#     x = vectorizor.transform([item.summary])
#     return max(regressor.predict(x)[0], 0)

# evaluate(natural_language_linear_regression_pricer, test)


# --------------------------------------------------------------

subset = 15_000
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf_model.fit(X[:subset], prices[:subset])

def random_forest(item):
    x = vectorizor.transform([item.summary])
    return max(0, rf_model.predict(x)[0])

evaluate(random_forest, test)