"""
utils/llm_utils.py
Llama a llama-3.3-70b (via Groq, que lo sirve gratis/rapido) para que
explique en lenguaje natural el resultado de una clasificacion.

IMPORTANTE: la variable de entorno debe llamarse SIEMPRE igual en todo
el proyecto -> GROQ_API_KEY. Ponla en un archivo .env en la raiz:
    GROQ_API_KEY=tu_api_key_aqui
"""
import os
from groq import Groq

# Se instancia una sola vez. Si no hay API key, se cae con un error claro
# en vez de fallar silenciosamente mas adelante.
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No se encontro GROQ_API_KEY en el entorno. "
                "Revisa tu archivo .env (el nombre de la variable debe ser identico)."
            )
        _client = Groq(api_key=api_key)
    return _client


def explicar_clasificacion(clase_predicha, atributos: dict, contexto_dataset: str = ""):
    """
    clase_predicha: valor que regreso el modelo (ej. 'versicolor')
    atributos: dict con los atributos que se usaron para clasificar
    contexto_dataset: opcional, texto libre describiendo el dataset
    """
    client = _get_client()

    atributos_txt = "\n".join(f"- {k}: {v}" for k, v in atributos.items())

    prompt = f"""Eres un asistente que explica resultados de un modelo de
clasificacion de machine learning a un usuario no tecnico.

Dataset: {contexto_dataset or "sin descripcion adicional"}

Atributos del ejemplo evaluado:
{atributos_txt}

Clase predicha por el modelo: {clase_predicha}

Explica en 3-4 oraciones, en espanol, por que este ejemplo probablemente
pertenece a esa clase, basandote en los valores de los atributos.
No inventes atributos que no te di."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return completion.choices[0].message.content
