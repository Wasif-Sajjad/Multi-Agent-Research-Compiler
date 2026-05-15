@echo off
:: This sets the Python path to the current directory so modules like 'api' and 'graph' can be found
set PYTHONPATH=%cd%
echo Starting Multi-Agent Research Compiler UI...
.\venv\Scripts\streamlit.exe run ui\app.py %*
