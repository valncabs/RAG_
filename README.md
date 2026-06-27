Perfecto, te dejo el **README listo para copiar y pegar tal cual en tu proyecto** 👇

---

# 📦 Proyecto RAG (Retrieval Augmented Generation)

Este proyecto implementa un sistema RAG usando Python, LangChain y FastAPI. Permite cargar documentos, procesarlos y consultarlos mediante una API.

---

## ⚠️ Requisitos previos

Antes de iniciar, asegúrate de tener instalado:

* Python 3.10 o 3.11 (recomendado ⚠️ NO usar 3.13)
* pip
* PowerShell (Windows)

---

## 📁 1. Clonar o ubicar el proyecto

Ubícate en la carpeta del proyecto:

```bash
cd C:\Users\tunombre\Desktop\RAG
```

---

## 🧪 2. Crear entorno virtual

Instala virtualenv (solo la primera vez):

```bash
pip install virtualenv
```

Crea el entorno virtual:

```bash
python -m venv myvenv
```

Esto generará la carpeta `myvenv`.

---

## ⚡ 3. Activar entorno virtual

En PowerShell ejecuta:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Luego activa el entorno:

```bash
.\myvenv\Scripts\Activate
```

Si se activó correctamente verás:

```bash
(myvenv)
```

---

## 📦 4. Instalación de dependencias

Con el entorno activado, instala las librerías necesarias:

```bash
pip install langchain langchain-community langchain-huggingface
pip install chromadb
pip install huggingface_hub
pip install pypdf
pip install python-dotenv
pip install sentence-transformers
pip install fastapi "uvicorn[standard]"
pip install pydantic
pip install langchain-text-splitters
pip install langchain-core
pip install langchain-chroma
pip install langchain-groq
```

---

## 🚀 5. Ejecutar el proyecto

### 📌 Paso 1: Ingesta de datos

```bash
python -m app.ingest
```

### 📌 Paso 2: Levantar servidor

```bash
uvicorn app.main:app --reload
```

---

## 🌐 6. Probar la API

Abre en tu navegador:

```
http://127.0.0.1:8000/docs
```

Aquí podrás probar los endpoints de la API.

---

## 🧠 Notas importantes

* Siempre activa el entorno virtual antes de ejecutar el proyecto:

  ```bash
  .\myvenv\Scripts\Activate
  ```
* Se recomienda usar Python 3.10 o 3.11 para evitar errores de compatibilidad.
* Si ocurre algún error, guarda el mensaje completo.

---

## 🛠️ Estructura del proyecto

```
RAG/
│
├── app/
├── myvenv/
├── docs/
├── chroma_db/ <--- se genera solo al ejecutar comando (python -m app.ingest)
├── .env
```

---


