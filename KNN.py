import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel(r"C:\Users\gokay\Desktop\Makaleler\verisetiorjinal.xlsx")

data.ffill(inplace=True)

data = data[data['Destination'] != 'Unknown']

X = data[['Traveler age', 'Traveler gender', 'Traveler nationality',
          'Accommodation type', 'Accommodation cost', 'Transportation type', 'Transportation cost']].copy()
y = data['Destination'].copy()

label_encoders = {}
for column in ['Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

print("Please answer the following questions:")
traveler_age = int(input("Enter the traveler's age: "))
traveler_gender = input("Enter the traveler's gender (Male/Female): ")
traveler_nationality = input("Enter the traveler's nationality: ")
accommodation_type = input("Enter the accommodation type (Hotel/House/Camp etc.): ")
accommodation_cost = int(input("Enter the accommodation cost: "))
transportation_type = input("Enter the transportation type (Plane/Train/Bus etc.): ")
transportation_cost = int(input("Enter the transportation cost: "))

new_traveler = pd.DataFrame({'Traveler age': [traveler_age], 'Traveler gender': [traveler_gender],
                             'Traveler nationality': [traveler_nationality], 'Accommodation type': [accommodation_type],
                             'Accommodation cost': [accommodation_cost], 'Transportation type': [transportation_type],
                             'Transportation cost': [transportation_cost]})

for column in ['Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']:
    new_traveler[column] = new_traveler[column].apply(lambda x: label_encoders[column].transform([x])[0]
                                                      if x in label_encoders[column].classes_ else -1)
    if -1 in new_traveler[column].values:
        raise ValueError(f"'{column}' contains previously unseen labels.")

predicted_destination = knn.predict(new_traveler)
print("Predicted destination:", predicted_destination[0])
