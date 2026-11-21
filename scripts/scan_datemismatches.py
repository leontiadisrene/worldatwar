name: Run Date Mismatch Diagnostic

on:
  # Allows you to trigger this workflow manually from the GitHub Actions tab
  workflow_dispatch:

jobs:
  run-diagnostic:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.10"

      - name: Run date mismatch diagnostic
        working-directory: scripts
        run: python scan_datemismatches.py
