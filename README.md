# ACC Spring 2026 LF Project

## Commands
Create branch:
```bash
git checkout -b branch-name
```

Go into existing branch:
```bash
git checkout branch-name
```

Git track your changes:
```bash
git add filepath
```
Or to add all files
```bash
git add .
```
Then start tracking
```bash
git commit -m "Description of changes"
```

Open a pull request (after commiting files):
```bash
git push origin branch-name
```
Then go to github and there should be a popup to open the pull request

Create python environment (Bash):
```bash
conda create -n MLOpsEnv python=3.11
conda activate MLOpsEnv
pip install -r requirements.txt
```
Create python environment (Powershell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```