import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# ================= LOAD DATA =================
data = pd.read_csv("../data/flood_data.csv")

# ================= FEATURE ENGINEERING =================
data["rain_intensity"] = data["rainfall_mm"] / 24
data["heat_index"] = (data["temperature"] * data["humidity"]) / 100
data["moisture_index"] = (data["rainfall_mm"] * data["humidity"]) / 100

# Proxy for slope instability (hackathon-acceptable)
data["soil_saturation"] = data["rainfall_mm"] + data["moisture_index"]

# ================= LABEL ENGINEERING =================
# FLOOD RISK
def flood_risk(row):
    if row["rainfall_mm"] > 150 and row["moisture_index"] > 80:
        return 2
    elif row["rainfall_mm"] > 80:
        return 1
    return 0

# LANDSLIDE RISK
def landslide_risk(row):
    if row["soil_saturation"] > 200:
        return 2
    elif row["soil_saturation"] > 120:
        return 1
    return 0

# HEATWAVE RISK
def heatwave_risk(row):
    if row["temperature"] > 40 and row["humidity"] > 60:
        return 2
    elif row["temperature"] > 35:
        return 1
    return 0

data["flood_risk"] = data.apply(flood_risk, axis=1)
data["landslide_risk"] = data.apply(landslide_risk, axis=1)
data["heatwave_risk"] = data.apply(heatwave_risk, axis=1)

# ================= FEATURES =================
features = [
    "rainfall_mm",
    "temperature",
    "humidity",
    "rain_intensity",
    "heat_index",
    "moisture_index",
    "soil_saturation"
]

X = data[features]
y = data[["flood_risk", "landslide_risk", "heatwave_risk"]]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ================= MODEL =================
base_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)

model = MultiOutputClassifier(base_model)

# ================= TRAIN =================
model.fit(X_train, y_train)

# ================= EVALUATE =================
y_pred = model.predict(X_test)

print("\nFLOOD RISK REPORT\n")
print(classification_report(y_test["flood_risk"], y_pred[:, 0]))

print("\nLANDSLIDE RISK REPORT\n")
print(classification_report(y_test["landslide_risk"], y_pred[:, 1]))

print("\nHEATWAVE RISK REPORT\n")
print(classification_report(y_test["heatwave_risk"], y_pred[:, 2]))

# ================= SAVE =================
joblib.dump(model, "multi_hazard_model.pkl")
print("\nMulti-hazard model saved as multi_hazard_model.pkl")
