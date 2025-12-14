# -----------------------------------------------------------
#  RECOMENDACIÓN — TF-IDF + COSENO
# -----------------------------------------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar dataset
df = pd.read_csv("cleaned_data-v1.csv", encoding="latin1")

# Asegurar columna
if "DESCRIPCIÓN_LIMPIA" not in df.columns:
    raise KeyError("La columna DESCRIPCIÓN_LIMPIA no existe en el dataset.")

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words="spanish")
tfidf_matrix = vectorizer.fit_transform(df["DESCRIPCIÓN_LIMPIA"].fillna(""))

# Función de recomendación
def recomendar_medicamentos(texto_busqueda, top_n=5):
    texto_vector = vectorizer.transform([texto_busqueda])
    similitudes = cosine_similarity(texto_vector, tfidf_matrix).flatten()

    indices_top = similitudes.argsort()[::-1][:top_n]

    return df.iloc[indices_top][["CODIGO", "DESCRIPCIÓN", "PRECIO_UNITARIO"]]

