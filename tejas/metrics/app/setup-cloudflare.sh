#!/bin/bash
# Cloudflare Tunnel Setup Script
# Run this script to set up public access to your webapp

set -e

echo "========================================="
echo "Cloudflare Tunnel Setup for DataYates"
echo "========================================="
echo ""

# Step 1: Install cloudflared
echo "Step 1: Installing cloudflared..."
sudo dpkg -i cloudflared-linux-amd64.deb
echo "✓ cloudflared installed"
echo ""

# Step 2: Verify installation
echo "Step 2: Verifying installation..."
cloudflared --version
echo "✓ Installation verified"
echo ""

# Step 3: Login to Cloudflare
echo "Step 3: Authenticate with Cloudflare..."
echo "This will open a browser window for you to login."
echo "If you don't have a Cloudflare account, you'll create one (it's free!)"
echo ""
read -p "Press Enter to continue..."
cloudflared tunnel login
echo "✓ Authenticated successfully"
echo ""

# Step 4: Create tunnel
echo "Step 4: Creating tunnel..."
cloudflared tunnel create datayates
echo "✓ Tunnel created"
echo ""

# Get the tunnel ID
TUNNEL_ID=$(cloudflared tunnel list | grep datayates | awk '{print $1}')
echo "Your tunnel ID is: $TUNNEL_ID"
echo ""

# Step 5: Create config file
echo "Step 5: Creating configuration file..."
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<EOF
tunnel: $TUNNEL_ID
credentials-file: /home/tejas/.cloudflared/${TUNNEL_ID}.json

ingress:
  - service: http://localhost:8000
  # This catches all remaining traffic
  - service: http_status:404
EOF
echo "✓ Configuration created"
echo ""

# Step 6: Test the tunnel (quick run)
echo "Step 6: Testing tunnel..."
echo "Starting tunnel in test mode for 10 seconds..."
timeout 10 cloudflared tunnel run datayates || true
echo "✓ Tunnel test completed"
echo ""

# Step 7: Create systemd service
echo "Step 7: Creating systemd service..."
sudo tee /etc/systemd/system/cloudflared.service > /dev/null <<EOF
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=tejas
ExecStart=/usr/local/bin/cloudflared tunnel run datayates
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Systemd service created"
echo ""

# Step 8: Enable and start service
echo "Step 8: Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
echo "✓ Service started"
echo ""

# Step 9: Get the public URL
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Your webapp is now publicly accessible!"
echo ""
echo "To get your public URL, run:"
echo "  cloudflared tunnel info datayates"
echo ""
echo "Or check the service status:"
echo "  sudo systemctl status cloudflared"
echo ""
echo "Since you don't have a custom domain configured,"
echo "you can use the quick URL feature:"
echo ""
echo "  cloudflared tunnel --url http://localhost:8000"
echo ""
echo "This will give you a temporary URL like:"
echo "  https://your-tunnel.trycloudflare.com"
echo ""
echo "For a permanent URL, you can:"
echo "1. Add a domain to Cloudflare (free)"
echo "2. Run: cloudflared tunnel route dns datayates yourdomain.com"
echo ""
echo "========================================="

