# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import json
import csv

# %%
url = "https://api.openweathermap.org/data/2.5/forecast?lat=41.499321&lon=-81.694359&appid=1b1572712a676a1b6e72bb7cdb538e3f"

response = requests.get(url)
data = json.loads(response.text)

with open("forecasts.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["date_time", "temperature", "feels_like", "temp_min", "temp_max", "pressure", "sea_level", "grnd_level", "humidity", "weather_description", "clouds_all", "wind_speed", "wind_deg"])
    for item in data["list"]:
        writer.writerow([
            item["dt_txt"],
            item["main"]["temp"],
            item["main"]["feels_like"],
            item["main"]["temp_min"],
            item["main"]["temp_max"],
            item["main"]["pressure"],
            item["main"].get("sea_level", ""),
            item["main"].get("grnd_level", ""),
            item["main"]["humidity"],
            item["weather"][0]["description"],
            item["clouds"]["all"],
            item["wind"]["speed"],
            item["wind"]["deg"]
        ])

# %%
from datetime import datetime

with open("forecasts.csv", mode="r") as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)
    data = list(reader)

date_groups = {}
for row in data:
    date_str = row[0].split()[0]
    if date_str not in date_groups:
        date_groups[date_str] = []
    date_groups[date_str].append(row)

selected_rows = []
for date_str, rows in date_groups.items():
    rows_sorted = sorted(rows, key=lambda row: datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
    selected_row = rows_sorted[0]
    selected_rows.append(selected_row)

with open("forecast2.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(selected_rows)

# %%


df = pd.read_csv("forecast2.csv")
df_test = df.drop("date_time", axis=1)
df_test_y = df_test["weather_description"]
df_test = df_test.drop("weather_description", axis=1)




# %%
data = pd.read_csv("cleveland_weather.csv", usecols=[ "temperature", "feels_like", "temp_min", "temp_max", "pressure","sea_level","grnd_level","humidity","clouds_all","wind_speed","wind_deg", "weather_conditions"])

X = data.drop("weather_conditions", axis=1)
X = X.fillna(-9999999)
y = data["weather_conditions"]


# %%
from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
rfe.fit(X, y)
rfe.ranking_



# %%
mask = rfe.support_
selected_cols = X.columns[mask]
selected_df = pd.DataFrame()
selected_test = pd.DataFrame()
df_testdf = pd.DataFrame(df_test)
for col in selected_cols:
    selected_df[col] = X[col].values
    selected_test[col] = df_testdf[col].values

# %%
selected_df.head()

# %%

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
X_resampled, y_resampled = oversample.fit_resample(selected_df, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree accuracy on train data:", accuracy)

y_pred = model.predict(selected_test)
accuracy = accuracy_score(df_test_y, y_pred)
print("Decision Tree accuracy on test:", accuracy)

new_prediction = model.predict(selected_test)
print("Decision Tree prediction of test data:", new_prediction)

# %%
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)
y_pred_rf = rfc.predict(X_test)
accuracy = rfc.score(X_test, y_test)
print("Random Forest Accuracy for train data: ",  accuracy)

y_pred_rf = rfc.predict(selected_test)
accuracy = rfc.score(selected_test, df_test_y)
print("Random Forest Accuracy for test data: " , accuracy)
print("Random Forest Prediction of : " , y_pred_rf)

# %%


print("Actual forecasts for the next 6 days is: light rain, light rain, few clouds, broken clouds, broken clouds, broken clouds")

# %%
import tkinter as tk
from tkinter import ttk

# %%
def kelvin_to_fahrenheit(temps):
        """Convert a list of temperatures from Kelvin to Fahrenheit."""
        return [((temp - 273.15) * 9/5) + 32 for temp in temps]

# %%
import tkinter as tk
import tkinter.ttk as ttk

class WeatherForecastGUI:
   

    def __init__(self, master):
        self.master = master
        master.title("Cleveland Weather Forecast")
        master.geometry("800x400")

        # create a table to display the weather data
        self.table = ttk.Treeview(master)
        
        self.table["columns"] = ( "temperature", "weather_conditions1", "weather_conditions2", "actual_weather_forecast" )
        
        self.table.heading("temperature", text="Temperature")
        self.table.heading("weather_conditions1", text="Prediction(Random_forest)")
        self.table.heading("weather_conditions2", text="Prediction(Decison_tree)")
        self.table.heading("actual_weather_forecast", text="Real Weather Forecast")
        # add the predicted weather data to the table
        temperature = df_test["temperature"]

        temperature =kelvin_to_fahrenheit(temperature)
        for temp in range(6):
            temperature[temp] = round(temperature[temp], 2)


        for i in range(6):
            day = "Day " + str(i+1)
            self.table.insert("", tk.END, text=day,  values=(temperature[i] , y_pred_rf[i], new_prediction[i], df_test_y[i]))

      
        self.table.column("temperature", width=100)
        self.table.column("weather_conditions1", width=200)
        self.table.column("weather_conditions2", width=200)
        self.table.column("actual_weather_forecast", width=200)
        # display the table
        self.table.pack()

    

if __name__ == '__main__':
    root = tk.Tk()
    WeatherForecastGUI(root)
    root.mainloop()



