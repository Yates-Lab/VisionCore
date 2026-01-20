"""
Simple FastAPI Hello World Application
This will eventually be expanded to serve data and images with interactive features.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(
    title="DataYates Web App",
    description="Data visualization and analysis web application",
    version="0.1.0"
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Simple hello world page
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataYates</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                text-align: center;
                padding: 40px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            }
            h1 {
                font-size: 3em;
                margin: 0;
                animation: fadeIn 1s ease-in;
            }
            p {
                font-size: 1.2em;
                margin-top: 20px;
                animation: fadeIn 1.5s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Hello World!</h1>
            <p>Welcome to DataYates Web Application</p>
            <p style="font-size: 0.9em; margin-top: 30px;">
                Server is running successfully ✓
            </p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "Server is running"}

@app.get("/api/info")
async def info():
    """
    API info endpoint
    """
    return {
        "app": "DataYates Web App",
        "version": "0.1.0",
        "description": "Data visualization and analysis platform"
    }

if __name__ == "__main__":
    # For development only - in production, use systemd + uvicorn directly
    uvicorn.run(app, host="0.0.0.0", port=8000)

