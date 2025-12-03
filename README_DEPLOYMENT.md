# 🚀 Deployment Guide

## 1. Prerequisites
-   VPS (AWS EC2, DigitalOcean, etc.)
-   Ubuntu 20.04+
-   Python 3.9+

## 2. Setup
```bash
# Clone Repo
git clone <repo_url>
cd crypto-ai-decision-system

# Install Dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configuration
-   Set API Keys in environment variables or `.env` (if using python-dotenv).
-   `export BINANCE_API_KEY="your_key"`
-   `export BINANCE_SECRET_KEY="your_secret"`

## 4. Run Manually
```bash
chmod +x deployment/run_bot.sh
./deployment/run_bot.sh
```

## 5. Run as Service (Systemd)
```bash
# Edit path/user in deployment/systemd_service.conf if needed
sudo cp deployment/systemd_service.conf /etc/systemd/system/crypto-bot.service
sudo systemctl daemon-reload
sudo systemctl enable crypto-bot
sudo systemctl start crypto-bot
sudo systemctl status crypto-bot
```

## 6. Monitoring
Run the dashboard:
```bash
streamlit run src/app/monitor_dashboard.py
```
Access at `http://<server-ip>:8501`.
