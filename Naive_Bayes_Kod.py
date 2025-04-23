import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_excel(r"C:\Users\Nese\Desktop\veribilimi.xlsx")

data.ffill(inplace=True)

data = data[data['Destination'] != 'Unknown']

X = data[['Traveler age', 'Traveler gender', 'Traveler nationality',
          'Accommodation type', 'Accommodation cost', 'Transportation type', 'Transportation cost']].copy()
y = data['Destination'].copy()

label_encoders = {}
for column in ['Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

nb = GaussianNB()
nb.fit(X, y)

print("Lütfen aşağıdaki soruları cevaplayın:")
traveler_age = int(input("Seyahat eden kişinin yaşını girin: "))
traveler_gender = input("Seyahat eden kişinin cinsiyetini girin (Erkek/Kadın): ")
traveler_nationality = input("Seyahat eden kişinin uyruğunu girin: ")
accommodation_type = input("Konaklama türünü girin (Otel/Ev/Kamp vb.): ")
accommodation_cost = int(input("Konaklama maliyetini girin: "))
transportation_type = input("Ulaşım türünü girin (Uçak/Tren/Otobüs vb.): ")
transportation_cost = int(input("Ulaşım maliyetini girin: "))

new_traveler = pd.DataFrame({'Traveler age': [traveler_age], 'Traveler gender': [traveler_gender],
                             'Traveler nationality': [traveler_nationality], 'Accommodation type': [accommodation_type],
                             'Accommodation cost': [accommodation_cost], 'Transportation type': [transportation_type],
                             'Transportation cost': [transportation_cost]})

for column in ['Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']:
    new_traveler[column] = new_traveler[column].apply(lambda x: label_encoders[column].transform([x])[0]
                                                      if x in label_encoders[column].classes_ else -1)
    if -1 in new_traveler[column].values:
        raise ValueError(f"'{column}' contains previously unseen labels.")

predicted_destination = nb.predict(new_traveler)
print("Tahmin edilen destinasyon:", predicted_destination[0])
