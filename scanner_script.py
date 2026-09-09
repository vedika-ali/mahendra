name: Automated Stock Paper Trader

on:
  schedule:
    - cron: '45 3 * * 1-5'  # Mon-Fri at 9:15 AM IST
  workflow_dispatch:

jobs:
  run-scanner:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install yfinance pandas numpy

    - name: Run Paper Trading Script
      run: |
        python scanner_script.py

    - name: Commit and Push Updated CSV
      run: |
        git config --global user.name "GitHub Actions Bot"
        git config --global user.email "actions@github.com"
        git add paper_trades.csv
        git diff --staged --quiet || git commit -m "Auto-update paper trades CSV [skip ci]"
        git push
        df.to_csv('paper_trades.csv', index=False)
