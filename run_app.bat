@echo off
setlocal

start cmd /k "cd app\backend && call venv_backend\Scripts\activate && uvicorn main:app --reload"
start cmd /k "cd app\frontend && call venv_frontend\Scripts\activate && streamlit run frontend_main.py"

echo Both servers are running...
endlocal
