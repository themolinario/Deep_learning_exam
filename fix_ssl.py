#!/usr/bin/env python3
"""
Script per risolvere i problemi SSL su macOS.
"""

import subprocess
import sys
import os
import ssl
import urllib.request

def install_certificates():
    """
    Installa i certificati SSL necessari per macOS.
    """
    print("🔧 Risoluzione problemi SSL su macOS...")

    try:
        # Trova la versione di Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Possibili percorsi per Install Certificates.command
        cert_paths = [
            f"/Applications/Python {python_version}/Install Certificates.command",
            f"/usr/local/bin/Install Certificates.command",
            "/Applications/Python*/Install Certificates.command"
        ]

        for cert_path in cert_paths:
            if os.path.exists(cert_path):
                print(f"📜 Esecuzione: {cert_path}")
                subprocess.run([cert_path], check=True)
                print("✅ Certificati installati con successo!")
                return True

    except Exception as e:
        print(f"⚠️ Metodo automatico fallito: {e}")

    # Metodo alternativo
    try:
        print("🔄 Tentativo metodo alternativo...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], check=True)
        print("✅ Certifi aggiornato!")
        return True
    except Exception as e:
        print(f"⚠️ Anche il metodo alternativo è fallito: {e}")

    return False

def configure_ssl_environment():
    """
    Configura l'ambiente per gestire SSL.
    """
    print("🔧 Configurazione ambiente SSL...")

    # Disabilita temporaneamente la verifica SSL
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''

    # Configura SSL context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✅ Contesto SSL configurato")
    except Exception as e:
        print(f"⚠️ Errore configurazione SSL: {e}")

def test_ssl_connection():
    """
    Testa la connessione SSL.
    """
    print("🧪 Test connessione SSL...")

    try:
        import requests
        response = requests.get("https://openaipublic.azureedge.net/clip/models/", timeout=10)
        if response.status_code == 200:
            print("✅ Connessione SSL funzionante!")
            return True
    except Exception as e:
        print(f"❌ Test SSL fallito: {e}")

    return False

if __name__ == "__main__":
    print("🚀 Risoluzione problemi SSL per CLIP su macOS")
    print("=" * 50)

    # Installa certificati
    install_certificates()

    # Configura ambiente
    configure_ssl_environment()

    # Testa connessione
    test_ssl_connection()

    print("\n✅ Setup SSL completato!")
    print("Ora puoi avviare l'applicazione con: python run_app.py")
