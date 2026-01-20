# DataYates Web Application

A FastAPI-based web application for data visualization and analysis, accessible over the internet.

## Stack

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **systemd**: Process manager (keeps app running 24/7)
- **Nginx**: Reverse proxy and web server
- **Let's Encrypt**: SSL certificates (optional, for HTTPS)

## Setup Instructions

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and Nginx
sudo apt install python3 python3-pip python3-venv nginx -y
```

### 2. Set Up Conda Environment

```bash
# Navigate to app directory
cd /home/tejas/DataYatesV1/tejas/metrics/app

# Activate your conda environment
conda activate env

# Install dependencies
pip install -r requirements.txt

# OR if you prefer conda packages:
# conda install fastapi uvicorn
```

**Note**: If you want to use a different conda environment name, update the `datayates-webapp.service` file accordingly.

### 3. Test the Application Locally

```bash
# Make sure you're in the app directory with conda env activated
conda activate env
python main.py

# OR use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000

# Test in another terminal:
curl http://localhost:8000
curl http://localhost:8000/api/health
```

Visit `http://localhost:8000` in your browser to see the hello world page.

### 4. Set Up systemd Service (Keep it Running)

**Important**: Before copying the service file, verify your conda installation path:

```bash
# Find your conda path
which conda
# This will show something like: /home/tejas/miniconda3/bin/conda
# OR: /home/tejas/anaconda3/bin/conda

# Verify your conda environment exists
conda env list
```

If your conda path is different from `/home/tejas/miniconda3`, edit `datayates-webapp.service` and update the paths in the `ExecStart` line.

Then set up the service:

```bash
# Copy service file to systemd directory
sudo cp datayates-webapp.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable datayates-webapp

# Start the service
sudo systemctl start datayates-webapp

# Check status
sudo systemctl status datayates-webapp
```

**Useful systemd commands:**
```bash
# View logs
sudo journalctl -u datayates-webapp -f

# Restart service
sudo systemctl restart datayates-webapp

# Stop service
sudo systemctl stop datayates-webapp

# Disable service (won't start on boot)
sudo systemctl disable datayates-webapp
```

### 5. Configure Nginx

```bash
# Copy nginx config
sudo cp nginx-datayates.conf /etc/nginx/sites-available/datayates

# Create symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/datayates /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# If test passes, restart nginx
sudo systemctl restart nginx

# Enable nginx to start on boot
sudo systemctl enable nginx
```

### 6. Configure Firewall

```bash
# Allow HTTP and HTTPS through firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check firewall status
sudo ufw status

# If firewall is not enabled, enable it (be careful with SSH access!)
# sudo ufw allow 22/tcp  # Allow SSH first!
# sudo ufw enable
```

### 7. Network/Router Configuration

If this machine is behind a router (NAT), you'll need to:

1. **Port Forwarding**: Configure your router to forward ports 80 and 443 to this machine's local IP
2. **Static IP**: Consider setting a static local IP for this machine on your router
3. **Dynamic DNS** (optional): If your public IP changes, use a service like DuckDNS or No-IP

### 8. Access Your Application

Once everything is set up:

- **Local network**: `http://192.168.x.x` or `http://localhost`
- **Internet**: `http://169.229.228.105` (your public IP)

Test with:
```bash
curl http://169.229.228.105
curl http://169.229.228.105/api/health
```

## Optional: Set Up HTTPS with Let's Encrypt

**Important**: You'll need a domain name for Let's Encrypt. If you don't have one, you can:
- Use a free service like FreeDNS, DuckDNS, or No-IP
- Buy a cheap domain (~$10/year)

Once you have a domain:

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate (replace YOUR_DOMAIN)
sudo certbot --nginx -d YOUR_DOMAIN

# Certbot will automatically configure nginx for HTTPS
# Certificates auto-renew via cron

# Test renewal
sudo certbot renew --dry-run
```

If you want to use IP address only (no domain), you can skip HTTPS for now or use a self-signed certificate.

## Development vs Production

**Development** (for testing):
```bash
# Run directly
cd /home/tejas/DataYatesV1/tejas/metrics/app
conda activate env
python main.py
# OR
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production** (for public access):
- Use systemd service (always running, auto-restarts)
- Use Nginx (handles SSL, static files, security)
- No `--reload` flag

## Architecture

```
Internet
    ↓
Router (port forwarding 80/443)
    ↓
Nginx (port 80/443)
    ↓
FastAPI via Uvicorn (port 8000, localhost only)
    ↓
Your Python Application
```

## Monitoring and Maintenance

```bash
# Check if app is running
sudo systemctl status datayates-webapp

# Check nginx
sudo systemctl status nginx

# View application logs
sudo journalctl -u datayates-webapp -n 100

# View nginx logs
sudo tail -f /var/log/nginx/datayates_access.log
sudo tail -f /var/log/nginx/datayates_error.log

# Check which processes are using port 8000
sudo lsof -i :8000

# Check which processes are using port 80
sudo lsof -i :80
```

## Troubleshooting

### App won't start
```bash
# Check logs
sudo journalctl -u datayates-webapp -n 50

# Check if port 8000 is already in use
sudo lsof -i :8000

# Try running manually to see errors
cd /home/tejas/DataYatesV1/tejas/metrics/app
conda activate env
uvicorn main:app --host 0.0.0.0 --port 8000

# If conda activation fails in systemd, check the conda path:
which conda
conda env list
```

### Nginx won't start
```bash
# Check nginx configuration
sudo nginx -t

# Check nginx logs
sudo journalctl -u nginx -n 50
```

### Can't access from internet
1. Check if firewall allows port 80/443
2. Check if router has port forwarding configured
3. Check if your public IP is correct: `curl ifconfig.me`
4. Check if nginx is running: `sudo systemctl status nginx`
5. Check if app is running: `sudo systemctl status datayates-webapp`

## Next Steps

Once this hello world is working, you can expand it to:

1. **Add data endpoints**: Serve your datasets via API
2. **Image serving**: Add routes to serve images
3. **Interactive features**: Add threshold controls, filtering, etc.
4. **Frontend**: Add React/Vue or keep it simple with templates
5. **Database**: Add PostgreSQL or SQLite for data storage
6. **Authentication**: Add user login if needed
7. **WebSockets**: For real-time data updates

## Security Considerations

Since this will be public:

1. **Keep software updated**: `sudo apt update && sudo apt upgrade`
2. **Use HTTPS**: Set up SSL certificates
3. **Firewall**: Only open necessary ports
4. **Rate limiting**: Add nginx rate limiting for APIs
5. **Input validation**: Validate all user inputs
6. **Monitoring**: Set up monitoring/alerting
7. **Backups**: Regular backups of data
8. **SSH security**: Use key-based auth, disable password auth

## API Documentation

FastAPI provides automatic interactive API docs:

- **Swagger UI**: `http://your-ip/docs`
- **ReDoc**: `http://your-ip/redoc`

## Files Overview

- `main.py`: FastAPI application
- `requirements.txt`: Python dependencies
- `datayates-webapp.service`: systemd service file
- `nginx-datayates.conf`: Nginx configuration
- `README.md`: This file

