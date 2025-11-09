@echo off
title Iniciar Frontend y Backend
echo Iniciando Backend...
start "Backend" cmd /k "python -m venv venv && call venv\Scripts\activate && uvicorn backend.api.app:app --reload"

timeout /t 5 >nul

echo Iniciando Frontend...
start "Frontend" cmd /k "cd frontend && npm install && npm start"

echo Todo iniciado. Cierra las ventanas para terminar.
pause
