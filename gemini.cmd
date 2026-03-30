@echo off
set "SCRIPT_DIR=%~dp0"
set "NODE_EXE=%SCRIPT_DIR%.node\node-v20.20.0-win-x64\node.exe"
set "PATH=%SCRIPT_DIR%.node\node-v20.20.0-win-x64;%PATH%"
"%NODE_EXE%" --no-warnings=DEP0040 "%SCRIPT_DIR%node_modules\@google\gemini-cli\dist\index.js" %*
