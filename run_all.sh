#!/bin/bash
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
poetry run python src/ui/gradio_app.py &
poetry run streamlit run src/ui/streamlit_app.py