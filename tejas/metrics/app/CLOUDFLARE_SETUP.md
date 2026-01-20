# Cloudflare Tunnel Setup Guide

Two options to make your webapp public:

## Option 1: Quick URL (Fastest - 2 minutes!)

This gives you an instant public URL without any setup.

**Run this command:**
```bash
# Install cloudflared first
cd /home/tejas/DataYatesV1/tejas/metrics/app
sudo dpkg -i cloudflared-linux-amd64.deb

# Start tunnel with quick URL
cloudflared tunnel --url http://localhost:8000
```

You'll get a URL like: `https://random-words-1234.trycloudflare.com`

**Pros:**
- ✅ Instant (no configuration needed)
- ✅ Free
- ✅ HTTPS automatically

**Cons:**
- ⚠️ URL changes each time you restart
- ⚠️ Tunnel stops when you close the terminal (unless you run in background)

**To run in background:**
```bash
nohup cloudflared tunnel --url http://localhost:8000 > cloudflare-tunnel.log 2>&1 &

# Check the URL in the log:
cat cloudflare-tunnel.log
```

---

## Option 2: Permanent Tunnel (Best for Long-Term)

This gives you a permanent setup that starts on boot.

**Run the setup script:**
```bash
cd /home/tejas/DataYatesV1/tejas/metrics/app
./setup-cloudflare.sh
```

The script will:
1. Install cloudflared
2. Help you login to Cloudflare (free account)
3. Create a tunnel named "datayates"
4. Set up systemd service (auto-start on boot)

**After setup, to get a public URL you need a domain:**

### Option A: Use Cloudflare's Free Subdomain
```bash
# Run the tunnel with quick URL even after setup
cloudflared tunnel --url http://localhost:8000
```

### Option B: Use Your Own Domain (Free if you have one)
```bash
# 1. Add your domain to Cloudflare (free)
# 2. Route traffic to your tunnel:
cloudflared tunnel route dns datayates yourdomain.com

# Now your app is at: https://yourdomain.com
```

---

## Management Commands

### Check tunnel status
```bash
sudo systemctl status cloudflared
```

### View tunnel logs
```bash
sudo journalctl -u cloudflared -f
```

### Restart tunnel
```bash
sudo systemctl restart cloudflared
```

### Stop tunnel
```bash
sudo systemctl stop cloudflared
```

### List all your tunnels
```bash
cloudflared tunnel list
```

### Get tunnel info
```bash
cloudflared tunnel info datayates
```

---

## Recommendation

**For immediate testing:** Use Option 1 (Quick URL)

**For production:** Use Option 2 (Permanent Tunnel) + your own domain

---

## Architecture

```
Internet Users
    ↓
Cloudflare CDN (global network)
    ↓ (encrypted tunnel)
UCSD Firewall (bypassed via outbound connection)
    ↓
Your Server (yoru)
    ↓
FastAPI App (port 8000)
```

The tunnel makes an **outbound** connection from your server to Cloudflare, which bypasses the university firewall that blocks **inbound** connections.

---

## Security Notes

- ✅ All traffic is encrypted (HTTPS automatically)
- ✅ Cloudflare provides DDoS protection
- ✅ No need to open firewall ports
- ✅ Rate limiting included
- ⚠️ Your app is now accessible to the entire internet
- ⚠️ Consider adding authentication if handling sensitive data

---

## Troubleshooting

### Tunnel won't start
```bash
# Check if cloudflared is installed
cloudflared --version

# Check if the app is running
curl http://localhost:8000
```

### Can't login to Cloudflare
Make sure you can access the browser window. If you're SSHed in, the login URL will be printed in the terminal - copy and paste it into a browser on your laptop.

### URL not working
Check if the tunnel is running:
```bash
sudo systemctl status cloudflared

# Or if using quick URL:
ps aux | grep cloudflared
```

---

## Next Steps

Once your app is public, you can:
1. Share the URL with collaborators
2. Add custom domain for a professional look
3. Build out your data visualization features
4. Add authentication for sensitive data
5. Set up monitoring/analytics

Enjoy your public webapp! 🎉

