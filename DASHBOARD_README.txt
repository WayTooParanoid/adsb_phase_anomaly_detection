ADS-B Phase DBSCAN Dashboard

Run:
  start_dashboard.cmd

Requirements:
  - Windows
  - Python 3.11 or newer on PATH
  - Internet access on first run so Python dependencies can be installed
  - Internet access while running live mode for ADS-B polling and map tiles

What the launcher does:
  - Creates a local .venv folder inside this project if needed
  - Installs requirements.txt into that local environment on first run
  - Starts the dashboard on http://127.0.0.1:8050
  - Opens the dashboard in the default browser

The trained DBSCAN models are included under artifacts\models.
