@echo off
start "Uvicorn" poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
start "Gradio" poetry run python src/ui/gradio_app.py
start "Streamlit" poetry run streamlit run src/ui/streamlit_app.py