@echo off
setlocal

echo Setting up backend environment...
if not exist "app\backend\venv_backend" (
    python -m venv app\backend\venv_backend
)
echo Installing backend dependencies...
call app\backend\venv_backend\Scripts\activate
pip install -r app\backend\requirements.txt
deactivate

echo Setting up frontend environment...
if not exist "app\frontend\venv_frontend" (
    python -m venv app\frontend\venv_frontend
)
echo Installing frontend dependencies...
call app\frontend\venv_frontend\Scripts\activate
pip install -r app\frontend\requirements.txt
deactivate

echo Setup complete.
endlocal
