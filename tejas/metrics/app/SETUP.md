# DataYates Webapp - Complete Setup & Management Guide

**Self-contained guide for starting, stopping, and managing a publicly accessible FastAPI web application running on a UCSD server.**

---

## 📋 System Overview

### What This Is

A FastAPI web application running on server `yoru` at UCSD, made publicly accessible via Cloudflare Tunnel (which bypasses the university firewall).

### Components

1. **FastAPI Application** - Python web server (main application)
2. **Cloudflare Tunnel** - Creates secure tunnel for public internet access
3. **Nginx** - Reverse proxy (currently not used with Cloudflare, but configured)
4. **systemd** - Linux service manager (keeps services running)

### File Locations

```
Application Directory: /home/tejas/DataYatesV1/tejas/metrics/app/
Main Application: /home/tejas/DataYatesV1/tejas/metrics/app/main.py
Requirements: /home/tejas/DataYatesV1/tejas/metrics/app/requirements.txt
Conda Environment: env (located at /home/tejas/.conda/envs/env)
```

### Services

```
FastAPI Service: datayates-webapp
Cloudflare Service: cloudflared (if using permanent setup)
Nginx Service: nginx
```

### Ports

```
FastAPI App: localhost:8000
Nginx: localhost:80 (not currently used)
Public Access: via Cloudflare tunnel (https://xxx.trycloudflare.com)
```

### Architecture

```
Internet Users
    ↓
Cloudflare CDN (global network)
    ↓ (encrypted tunnel - bypasses UCSD firewall)
Cloudflare Tunnel (cloudflared) running on yoru
    ↓
FastAPI App (uvicorn) on localhost:8000
    ↓
Python Application (main.py)
```

**Important:** Nginx is NOT in the chain when using Cloudflare Tunnel. The tunnel connects directly to port 8000.

---

## 🚀 STARTING THE WEBAPP

### Method 1: Quick Start (Public Access - Temporary URL)

Use this for testing or one-time use. The URL changes each time.

```bash
# Step 1: Start the FastAPI application
sudo systemctl start datayates-webapp

# Step 2: Verify it's running
sudo systemctl status datayates-webapp
# Should show "active (running)" in green

# Step 3: Start Cloudflare tunnel for public access
cloudflared tunnel --url http://localhost:8000

# Step 4: Look for the URL in the output
# You'll see a line like:
# "Your quick Tunnel has been created! Visit it at:"
# "https://random-words-1234.trycloudflare.com"

# Step 5: Copy that URL and share it - your app is now live!
```

**To stop this tunnel:** Press `Ctrl+C` in the terminal where cloudflared is running.

---

### Method 2: Start in Background (Temporary URL, Keeps Running)

Use this when you want to close the terminal but keep the tunnel running.

```bash
# Step 1: Start the FastAPI application
sudo systemctl start datayates-webapp

# Step 2: Start Cloudflare tunnel in background
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflare-tunnel.log 2>&1 &

# Step 3: Get the public URL from the log file
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com
# Look for a line like: https://random-words-1234.trycloudflare.com

# Step 4: Your app is now running in the background!
```

**To stop this tunnel:**
```bash
pkill -f cloudflared
```

---

### Method 3: Production Start (Permanent Setup, Auto-Restart)

Use this if you've run the `setup-cloudflare.sh` script for permanent setup.

```bash
# Start both services
sudo systemctl start datayates-webapp
sudo systemctl start cloudflared

# Enable auto-start on server reboot (optional but recommended)
sudo systemctl enable datayates-webapp
sudo systemctl enable cloudflared

# Check status
sudo systemctl status datayates-webapp
sudo systemctl status cloudflared
```

---

### Method 4: Local Only (No Public Access)

Use this for testing without making it public.

```bash
# Just start the FastAPI application
sudo systemctl start datayates-webapp

# Access at http://localhost:8000 (only on the server)
# Or if using Cursor/VS Code: http://localhost:8000 (auto port-forwarded to your laptop)
```

---

## 🛑 STOPPING THE WEBAPP

### Quick Stop (Stop Everything)

```bash
# Stop the FastAPI application
sudo systemctl stop datayates-webapp

# Stop Cloudflare tunnel
# If running as systemd service:
sudo systemctl stop cloudflared

# If running in background (nohup):
pkill -f cloudflared

# If running in terminal: Press Ctrl+C
```

---

### Stop Individual Components

```bash
# Stop only the FastAPI app (stops the webapp, but tunnel may still run)
sudo systemctl stop datayates-webapp

# Stop only the Cloudflare tunnel (app still runs locally)
sudo systemctl stop cloudflared
# OR
pkill -f cloudflared

# Stop nginx (optional - not currently used with Cloudflare)
sudo systemctl stop nginx
```

---

### Complete Shutdown + Disable Auto-Start

```bash
# Stop all services
sudo systemctl stop datayates-webapp
sudo systemctl stop cloudflared
sudo systemctl stop nginx
pkill -f cloudflared  # In case it's running in background

# Disable auto-start on boot
sudo systemctl disable datayates-webapp
sudo systemctl disable cloudflared
sudo systemctl disable nginx

# Verify everything is stopped
ps aux | grep -E "(uvicorn|cloudflared|nginx)" | grep -v grep
# If this returns nothing, everything is stopped
```

---

## ♻️ RESTARTING THE WEBAPP

### After Making Code Changes

If you edited `/home/tejas/DataYatesV1/tejas/metrics/app/main.py` or any Python files:

```bash
# Restart the FastAPI application to pick up changes
sudo systemctl restart datayates-webapp

# Verify it restarted successfully
sudo systemctl status datayates-webapp

# The Cloudflare tunnel doesn't need to restart
# Your public URL stays the same
```

---

### Restart Everything

```bash
# Restart all services
sudo systemctl restart datayates-webapp
sudo systemctl restart cloudflared  # Only if using systemd
sudo systemctl restart nginx         # Optional

# OR restart the background tunnel:
pkill -f cloudflared
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflare-tunnel.log 2>&1 &
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com  # Get new URL
```

**Note:** Restarting the tunnel (if using quick URL method) will give you a NEW public URL.

---

## 📊 CHECKING STATUS

### Check What's Running

```bash
# Check FastAPI app status
sudo systemctl status datayates-webapp

# Check Cloudflare tunnel status
sudo systemctl status cloudflared

# Check nginx status (optional)
sudo systemctl status nginx

# Check all at once
sudo systemctl status datayates-webapp cloudflared nginx
```

### Check If Services Are Actually Running

```bash
# Check if FastAPI is running on port 8000
sudo lsof -i :8000
# Should show: uvicorn with PID

# Check if nginx is running on port 80
sudo lsof -i :80
# Should show: nginx processes

# Check if cloudflared is running (any method)
ps aux | grep cloudflared | grep -v grep
# Should show: cloudflared process
```

### Get Your Public URL

```bash
# If using quick URL method (nohup):
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com

# If using permanent tunnel:
cloudflared tunnel info datayates

# If running in terminal: Look at the terminal output
```

### Test If Webapp Is Working

```bash
# Test locally on the server
curl http://localhost:8000/api/health
# Should return: {"status":"healthy","message":"Server is running"}

# Test the public URL (replace with your actual URL)
curl https://your-url.trycloudflare.com/api/health
```

---

## 🔍 VIEWING LOGS

### FastAPI Application Logs

```bash
# View live logs (press Ctrl+C to stop)
sudo journalctl -u datayates-webapp -f

# View last 50 lines
sudo journalctl -u datayates-webapp -n 50

# View last 100 lines
sudo journalctl -u datayates-webapp -n 100

# View logs from last hour
sudo journalctl -u datayates-webapp --since "1 hour ago"

# View logs from today
sudo journalctl -u datayates-webapp --since today
```

### Cloudflare Tunnel Logs

```bash
# If using systemd:
sudo journalctl -u cloudflared -f

# If using nohup (background):
tail -f /tmp/cloudflare-tunnel.log

# View full log:
cat /tmp/cloudflare-tunnel.log
```

### Nginx Logs (Optional)

```bash
# Access logs (who visited your site)
sudo tail -f /var/log/nginx/datayates_access.log

# Error logs
sudo tail -f /var/log/nginx/datayates_error.log
```

---

## 🔄 AUTO-START ON BOOT

### Enable Auto-Start (Recommended for Production)

Services will automatically start when the server boots up.

```bash
# Enable FastAPI app to start on boot
sudo systemctl enable datayates-webapp

# Enable Cloudflare tunnel to start on boot (if using permanent setup)
sudo systemctl enable cloudflared

# Enable nginx to start on boot (optional)
sudo systemctl enable nginx

# Verify they're enabled
systemctl is-enabled datayates-webapp
systemctl is-enabled cloudflared
systemctl is-enabled nginx
# Each should return: "enabled"
```

### Disable Auto-Start

Services will NOT start automatically on server reboot.

```bash
# Disable auto-start for each service
sudo systemctl disable datayates-webapp
sudo systemctl disable cloudflared
sudo systemctl disable nginx

# Verify they're disabled
systemctl is-enabled datayates-webapp
# Should return: "disabled"
```

---

## 🎯 COMMON SCENARIOS

### Scenario 1: "I want to test my app locally (no public access)"

```bash
sudo systemctl start datayates-webapp
# Access at: http://localhost:8000
```

### Scenario 2: "I want to show someone my app quickly (temporary public URL)"

```bash
sudo systemctl start datayates-webapp
cloudflared tunnel --url http://localhost:8000
# Copy the URL that appears and share it
# Press Ctrl+C when done
```

### Scenario 3: "I want to keep it running in the background"

```bash
sudo systemctl start datayates-webapp
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflare-tunnel.log 2>&1 &
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com
# Share the URL
```

### Scenario 4: "I made code changes and need to update the app"

```bash
# Edit your code in /home/tejas/DataYatesV1/tejas/metrics/app/main.py
# Then:
sudo systemctl restart datayates-webapp
# Your public URL stays the same (if tunnel is running)
```

### Scenario 5: "I want production setup (auto-start, permanent)"

```bash
# First, run the setup script (one-time):
cd /home/tejas/DataYatesV1/tejas/metrics/app
./setup-cloudflare.sh

# Then enable auto-start:
sudo systemctl enable datayates-webapp
sudo systemctl enable cloudflared

# Start services:
sudo systemctl start datayates-webapp
sudo systemctl start cloudflared
```

### Scenario 6: "I need to shut everything down"

```bash
sudo systemctl stop datayates-webapp
pkill -f cloudflared
# Everything is now stopped
```

### Scenario 7: "Server rebooted - how do I get it running again?"

```bash
# If services are enabled (auto-start):
# Just wait - they start automatically

# If services are NOT enabled:
sudo systemctl start datayates-webapp
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflare-tunnel.log 2>&1 &
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com
```

---

## 🐛 TROUBLESHOOTING

### Problem: "App won't start"

```bash
# Check why it failed
sudo journalctl -u datayates-webapp -n 50

# Check if port 8000 is already in use
sudo lsof -i :8000

# If something is using port 8000, kill it:
sudo kill <PID>  # Replace <PID> with the process ID from lsof

# Try starting again
sudo systemctl start datayates-webapp
```

### Problem: "Cloudflare tunnel won't connect"

```bash
# Check if cloudflared is installed
cloudflared --version
# Should show version number

# Check if app is running first (tunnel needs app to be running)
curl http://localhost:8000
# Should return HTML

# Try running tunnel manually to see errors
cloudflared tunnel --url http://localhost:8000
```

### Problem: "Can't access public URL"

```bash
# Step 1: Verify app is running
sudo systemctl status datayates-webapp
# Should be "active (running)"

# Step 2: Verify tunnel is running
ps aux | grep cloudflared | grep -v grep
# Should show cloudflared process

# Step 3: Test locally first
curl http://localhost:8000/api/health
# Should return: {"status":"healthy",...}

# Step 4: Get the correct URL
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com

# Step 5: Try the URL in an incognito browser window
```

### Problem: "Error: Address already in use (port 8000)"

```bash
# Find what's using port 8000
sudo lsof -i :8000

# You'll see output like:
# COMMAND   PID   USER
# python3  12345  tejas

# Kill the process (replace 12345 with actual PID)
sudo kill 12345

# Start the service again
sudo systemctl start datayates-webapp
```

### Problem: "After code changes, nothing happens"

```bash
# You must restart the service to pick up code changes
sudo systemctl restart datayates-webapp

# Verify it restarted
sudo systemctl status datayates-webapp
```

### Problem: "Cloudflared tunnel keeps disconnecting"

```bash
# Use systemd for automatic reconnection
sudo systemctl start cloudflared

# OR if you want auto-restart with nohup:
while true; do
  cloudflared tunnel --url http://localhost:8000
  sleep 5
done > /tmp/cloudflare-tunnel.log 2>&1 &
```

### Problem: "I forgot my public URL"

```bash
# If using nohup method:
cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com

# If using systemd:
sudo journalctl -u cloudflared -n 100 | grep trycloudflare.com

# If running in terminal: Check the terminal output
```

---

## 📝 QUICK REFERENCE

### Most Used Commands

| Task | Command |
|------|---------|
| Start app | `sudo systemctl start datayates-webapp` |
| Stop app | `sudo systemctl stop datayates-webapp` |
| Restart app | `sudo systemctl restart datayates-webapp` |
| Check app status | `sudo systemctl status datayates-webapp` |
| View app logs | `sudo journalctl -u datayates-webapp -f` |
| Start public tunnel | `cloudflared tunnel --url http://localhost:8000` |
| Start tunnel in background | `nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflare-tunnel.log 2>&1 &` |
| Stop tunnel | `pkill -f cloudflared` |
| Get public URL | `cat /tmp/cloudflare-tunnel.log \| grep trycloudflare.com` |
| Test locally | `curl http://localhost:8000/api/health` |

---

## ⚙️ CONFIGURATION FILES

### Main Application File

```
Location: /home/tejas/DataYatesV1/tejas/metrics/app/main.py
Purpose: FastAPI application code
Edit this to: Add new endpoints, modify behavior
After editing: sudo systemctl restart datayates-webapp
```

### Requirements File

```
Location: /home/tejas/DataYatesV1/tejas/metrics/app/requirements.txt
Purpose: Python package dependencies
Edit this to: Add new Python packages
After editing: 
  conda activate env
  pip install -r /home/tejas/DataYatesV1/tejas/metrics/app/requirements.txt
  sudo systemctl restart datayates-webapp
```

### Service Files

```
FastAPI Service: /etc/systemd/system/datayates-webapp.service
Cloudflare Service: /etc/systemd/system/cloudflared.service
Nginx Config: /etc/nginx/sites-available/datayates

After editing service files:
  sudo systemctl daemon-reload
  sudo systemctl restart <service-name>
```

---

## 🔐 IMPORTANT NOTES

### What MUST Be Running for Public Access

1. ✅ **datayates-webapp** (FastAPI app) - **REQUIRED**
   - This is your actual web application
   - Without this, nothing works

2. ✅ **cloudflared** (tunnel) - **REQUIRED for public access**
   - This makes your app accessible from the internet
   - Without this, app only works locally

3. ❌ **nginx** - **NOT REQUIRED** (currently unused)
   - Nginx is configured but not in the request path
   - Cloudflare tunnel connects directly to port 8000
   - Can be safely stopped to save resources

### URLs Explained

```
Local URL: http://localhost:8000
  - Only accessible from the server itself
  - Or via Cursor/VS Code port forwarding

Public URL: https://random-words-1234.trycloudflare.com
  - Accessible from anywhere on the internet
  - Changes each time you restart cloudflared (unless using permanent setup)
  - Get it from: cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com
```

### After Server Reboot

```
If services are ENABLED (auto-start):
  - They start automatically
  - Check status: sudo systemctl status datayates-webapp

If services are NOT ENABLED:
  - You must manually start them
  - Run: sudo systemctl start datayates-webapp
  - Run: cloudflared tunnel --url http://localhost:8000
```

### Security Considerations

```
✅ Your app is now publicly accessible to the entire internet
✅ Cloudflare provides DDoS protection
✅ All traffic is encrypted (HTTPS)
⚠️ Anyone with the URL can access your app
⚠️ Consider adding authentication for sensitive data
⚠️ Keep your software updated
```

---

## 🎓 NEXT STEPS

Once your webapp is running, you can:

1. **Add new endpoints** - Edit `main.py` to add features
2. **Serve data** - Add routes to display/download data
3. **Upload files** - Enable file uploads
4. **Add visualizations** - Create charts, graphs, interactive displays
5. **Add authentication** - Protect sensitive features
6. **Custom domain** - Set up a permanent, professional URL
7. **Database integration** - Store and query data persistently

### Example: Add a New Endpoint

1. Edit the file:
```bash
nano /home/tejas/DataYatesV1/tejas/metrics/app/main.py
```

2. Add this code before the `if __name__ == "__main__":` line:
```python
@app.get("/api/hello/{name}")
async def hello_name(name: str):
    return {"message": f"Hello, {name}!"}
```

3. Restart the app:
```bash
sudo systemctl restart datayates-webapp
```

4. Test it:
```bash
curl http://localhost:8000/api/hello/World
# Returns: {"message":"Hello, World!"}
```

5. Access from public URL:
```
https://your-url.trycloudflare.com/api/hello/World
```

---

## 📞 EMERGENCY COMMANDS

### "Everything is broken - start from scratch"

```bash
# Stop everything
sudo systemctl stop datayates-webapp
sudo systemctl stop cloudflared
pkill -f cloudflared

# Start fresh
sudo systemctl start datayates-webapp
cloudflared tunnel --url http://localhost:8000

# Get new URL and test
```

### "I need to free up port 8000 immediately"

```bash
sudo systemctl stop datayates-webapp
# Port 8000 is now free
```

### "I need to know if anything is running"

```bash
# Check all services
sudo systemctl status datayates-webapp cloudflared nginx

# Check all ports
sudo lsof -i :8000
sudo lsof -i :80

# Check all processes
ps aux | grep -E "(uvicorn|cloudflared|nginx)" | grep -v grep
```

---

## ✅ SUCCESS CHECKLIST

Before sharing your public URL, verify:

- [ ] FastAPI app is running: `sudo systemctl status datayates-webapp`
- [ ] Shows "active (running)": ✓
- [ ] Cloudflared tunnel is running: `ps aux | grep cloudflared | grep -v grep`
- [ ] Shows cloudflared process: ✓
- [ ] Local access works: `curl http://localhost:8000/api/health`
- [ ] Returns JSON with "healthy": ✓
- [ ] Public URL obtained: `cat /tmp/cloudflare-tunnel.log | grep trycloudflare.com`
- [ ] URL shows: https://xxx.trycloudflare.com ✓
- [ ] Public access works: Test URL in browser
- [ ] Page loads: ✓

If all checkboxes are checked, your webapp is live! 🎉

---

**End of Guide**

This document contains everything needed to manage the DataYates webapp. For questions about the code itself, see `main.py`. For Cloudflare tunnel advanced setup, see `CLOUDFLARE_SETUP.md`.
