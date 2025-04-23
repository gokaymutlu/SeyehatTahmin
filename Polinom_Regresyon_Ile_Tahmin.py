import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_excel(r"C:\Users\gokay\Desktop\veribilimi.xlsx")

data.fillna(method='ffill', inplace=True)

data = data[data['Destination'] != 'Unknown']

X_accommodation = data[['Traveler name', 'Traveler age', 'Traveler gender', 'Traveler nationality',
          'Accommodation cost', 'Transportation cost']].copy()
y_accommodation = data['Accommodation cost'].copy()

X_transportation = data[['Traveler name', 'Traveler age', 'Traveler gender', 'Traveler nationality',
          'Accommodation cost', 'Transportation cost']].copy()
y_transportation = data['Transportation cost'].copy()

label_encoders = {}
for column in X_accommodation.columns:
    label_encoders[column] = LabelEncoder()
    X_accommodation[column] = label_encoders[column].fit_transform(X_accommodation[column])
    X_transportation[column] = label_encoders[column].fit_transform(X_transportation[column])

X_train_accommodation, X_test_accommodation, y_train_accommodation, y_test_accommodation = train_test_split(X_accommodation, y_accommodation, test_size=0.2, random_state=42)
X_train_transportation, X_test_transportation, y_train_transportation, y_test_transportation = train_test_split(X_transportation, y_transportation, test_size=0.2, random_state=42)

poly_accommodation = PolynomialFeatures(degree=2)
X_train_accommodation_poly = poly_accommodation.fit_transform(X_train_accommodation)
X_test_accommodation_poly = poly_accommodation.transform(X_test_accommodation)

poly_transportation = PolynomialFeatures(degree=2)
X_train_transportation_poly = poly_transportation.fit_transform(X_train_transportation)
X_test_transportation_poly = poly_transportation.transform(X_test_transportation)

regressor_accommodation = LinearRegression()
regressor_accommodation.fit(X_train_accommodation_poly, y_train_accommodation)

regressor_transportation = LinearRegression()
regressor_transportation.fit(X_train_transportation_poly, y_train_transportation)

accuracy_accommodation = regressor_accommodation.score(X_test_accommodation_poly, y_test_accommodation)
accuracy_transportation = regressor_transportation.score(X_test_transportation_poly, y_test_transportation)
print("Konaklama Modeli Doğruluğu:", accuracy_accommodation)
print("Ulaşım Modeli Doğruluğu:", accuracy_transportation)

traveler_name = input("Lütfen seyahat eden kişinin adını girin: ")

selected_traveler_accommodation = data[data['Traveler name'] == traveler_name].iloc[0]
selected_traveler_features_accommodation = selected_traveler_accommodation[['Traveler name', 'Traveler age', 'Traveler gender', 'Traveler nationality',
                                                'Accommodation cost', 'Transportation cost']].copy()

for column in selected_traveler_features_accommodation.index:
    selected_traveler_features_accommodation[column] = label_encoders[column].transform([selected_traveler_features_accommodation[column]])[0]

selected_traveler_poly_accommodation = poly_accommodation.transform([selected_traveler_features_accommodation])
accommodation_cost_prediction = regressor_accommodation.predict(selected_traveler_poly_accommodation)[0]
print("Tahmini konaklama harcaması:", accommodation_cost_prediction)

selected_traveler_transportation = data[data['Traveler name'] == traveler_name].iloc[0]
selected_traveler_features_transportation = selected_traveler_transportation[['Traveler name', 'Traveler age', 'Traveler gender', 'Traveler nationality',
                                                'Accommodation cost', 'Transportation cost']].copy()

for column in selected_traveler_features_transportation.index:
    selected_traveler_features_transportation[column] = label_encoders[column].transform([selected_traveler_features_transportation[column]])[0]

selected_traveler_poly_transportation = poly_transportation.transform([selected_traveler_features_transportation])
transportation_cost_prediction = regressor_transportation.predict(selected_traveler_poly_transportation)[0]
print("Tahmini ulaşım harcaması:", transportation_cost_prediction)
