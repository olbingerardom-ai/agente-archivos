# Agente de Archivos (Streamlit)

Analiza tus archivos (CSV, XLSX, JSON, TXT, PDF) con preguntas en lenguaje natural y genera tablas, pivots y gráficas.

## Ejecutar en local
```bash
pip install -r requirements.txt
# Configura tu clave (PowerShell en Windows)
setx OPENAI_API_KEY "tu_api_key_aqui"
# macOS/Linux
export OPENAI_API_KEY="tu_api_key_aqui"

streamlit run app.py
```

## Desplegar en Streamlit Cloud
1. Sube este proyecto a un repositorio en GitHub.
2. Ve a https://share.streamlit.io/deploy e inicia sesión con GitHub.
3. Selecciona tu repo y el archivo `app.py`.
4. En **Settings → Secrets**, agrega:
```
OPENAI_API_KEY = tu_api_key_aqui
```
5. Pulsa **Deploy**. Obtendrás una URL pública.

## Notas
- El índice semántico usa `faiss-cpu` y embeddings de OpenAI (`text-embedding-3-small`).
- El modelo de orquestación por defecto es `gpt-4o-mini`. Puedes cambiarlo en la barra lateral.
- Para entornos sin conexión a OpenAI, puedes reemplazar embeddings por modelos locales (pendiente de variante).

---
Hecho con ❤️ por Kai.
