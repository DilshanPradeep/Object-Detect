"""
Object Detection Web App — Simple HTTPS Server
Run: python app.py
Access: https://localhost:8443 or https://<your-ip>:8443
"""

import http.server
import ssl
import os
import socket

PORT = 8443
HOST = "0.0.0.0"

def get_local_ip():
    """Get the local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def run():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cert = os.path.join(os.path.dirname(__file__), "cert.pem")
    key = os.path.join(os.path.dirname(__file__), "key.pem")

    # Generate self-signed cert if missing
    if not os.path.exists(cert) or not os.path.exists(key):
        print("Generating self-signed SSL certificate...")
        os.system(
            f'openssl req -newkey rsa:2048 -new -nodes -x509 '
            f'-days 365 -keyout "{key}" -out "{cert}" -subj "/CN=localhost"'
        )

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer((HOST, PORT), handler)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    local_ip = get_local_ip()
    print(f"\n  Object Detection Server Running\n")
    print(f"  Local:    https://localhost:{PORT}")
    print(f"  Network:  https://{local_ip}:{PORT}")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()

if __name__ == "__main__":
    run()
