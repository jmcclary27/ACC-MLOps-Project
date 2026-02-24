# ACC Spring 2026 LF Project

# Development Setup & Git Workflow Guide

This guide covers:

- Git branch workflow  
- Committing & pushing changes  
- Pull requests  
- Syncing with main  
- Fixing common mistakes  
- Python environment setup  

---

# Initial Setup (First Time Only)

Clone the repository:

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

Check your Git config:

```bash
git config --global user.name
git config --global user.email
```

If not set:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
```

---

# Branch Workflow (IMPORTANT)

Never work directly on `main`.

---

## Create a New Branch

```bash
git checkout -b feature/branch-name
```

Example:

```bash
git checkout -b feature/add-clause-classifier
```

---

## Switch to an Existing Branch

```bash
git checkout branch-name
```

---

## See Current Branch

```bash
git branch
```

The `*` shows your active branch.

---

## See All Branches (Local + Remote)

```bash
git branch -a
```

---

# Keeping Your Branch Updated

Before starting work each day:

```bash
git checkout main
git pull origin main
```

Then update your feature branch:

```bash
git checkout feature/branch-name
git merge main
```

Or (cleaner history):

```bash
git rebase main
```

---

# Tracking & Committing Changes

## See What Changed

```bash
git status
```

## View Differences

```bash
git diff
```

---

## Add Files to Tracking

Add a specific file:

```bash
git add filepath
```

Add everything:

```bash
git add .
```

---

## Commit Changes

```bash
git commit -m "Clear description of what you changed"
```

Good example:

```bash
git commit -m "Add training smoke test for clause classifier"
```

---

# Push to GitHub

Push your branch:

```bash
git push origin branch-name
```

First push? Use:

```bash
git push -u origin branch-name
```

Then go to GitHub → Open Pull Request.

---

# Pull Latest Changes (If Someone Updated Main)

If you're already on main:

```bash
git pull origin main
```

If you're on a feature branch and want latest updates:

```bash
git fetch origin
git merge origin/main
```

---

# Undo / Fix Mistakes

Unstage a file:

```bash
git restore --staged filepath
```

Undo changes to a file:

```bash
git restore filepath
```

Amend last commit:

```bash
git commit --amend
```

Reset last commit (keep changes):

```bash
git reset --soft HEAD~1
```

Reset last commit (delete changes):

```bash
git reset --hard HEAD~1
```

Be careful with `--hard`.

---

# Delete a Branch

Delete local branch:

```bash
git branch -d branch-name
```

Force delete:

```bash
git branch -D branch-name
```

Delete remote branch:

```bash
git push origin --delete branch-name
```

---

# View Commit History

```bash
git log
```

Compact version:

```bash
git log --oneline --graph --all
```

---

# VSCode Helpful Commands

Open VSCode in current folder:

```bash
code .
```

Open integrated terminal:

```
Ctrl + `
```

Select Python interpreter:

```
Ctrl + Shift + P → Python: Select Interpreter
```

---

# Python Environment Setup

Choose ONE method below.

---

## Option 1: Conda (Recommended for ML)

```bash
conda create -n MLOpsEnv python=3.11
conda activate MLOpsEnv
pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate:

```bash
conda deactivate
```

---

## Option 2: venv (Lightweight)

### Bash (Mac/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PowerShell (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate:

```bash
deactivate
```

---

# Formatting & Linting

```bash
ruff check .
ruff format .
```

---

# Installing New Packages

After installing a new package:

```bash
pip install package-name
pip freeze > requirements.txt
```

---

# Recommended Commit Workflow (Every Time)

1. `git checkout main`
2. `git pull origin main`
3. `git checkout -b feature/your-feature`
4. Write code
5. `git add .`
6. `git commit -m "Clear message"`
7. `git push origin feature/your-feature`
8. Open Pull Request