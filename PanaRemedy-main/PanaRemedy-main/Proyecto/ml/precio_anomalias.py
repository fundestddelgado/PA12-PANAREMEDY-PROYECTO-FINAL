# -----------------------------------------------------------
# DETECCIÓN DE ANOMALÍAS
# -----------------------------------------------------------
import pandas as pd
from sklearn.ensemble import IsolationForest

# Cargar dataset
df = pd.read_csv("cleaned_data-v1.csv", encoding="latin1")

columnas_requeridas = [
    "TOTAL DE EXISTENCIAS  DISPONIBLES ENERO 2024",
    "PRECIO_UNITARIO",
    "MONTO DE EXISTENCIAS EN B/."
]

# Validación de columnas
for col in columnas_requeridas:
    if col not in df.columns:
        raise KeyError(f"La columna requerida no existe: {col}")

# Extraer datos numéricos
X = df[columnas_requeridas].replace(",", "", regex=True).astype(float)

# Modelo Isolation Forest
modelo = IsolationForest(
    contamination=0.03, 
    random_state=42
)

modelo.fit(X)
df["anomalia"] = modelo.predict(X)

# -1 = anomalía, 1 = normal
anomalias = df[df["anomalia"] == -1]

