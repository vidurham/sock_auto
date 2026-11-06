@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
if not exist ".streamlit\secrets.toml" if "%APP_PASSWORD%"=="" (
  echo.
  echo No app password set. Copy .streamlit\secrets.toml.example to .streamlit\secrets.toml
  echo and put a password in it, then run this again.
  echo.
  pause
  exit /b 1
)
python -m streamlit run app.py
if errorlevel 1 pause
