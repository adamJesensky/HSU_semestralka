import requests
import time
import psutil    
import subprocess
from stem import Signal
from stem.control import Controller

# HOW to download and configure tor expert bundle
# https://www.youtube.com/watch?v=Ep5ytx8ad04
# treba ist do TOR zlozky C:\programs\Tor Browser\Browser\TorBrowser\Tor, pustit hore cmd cez adresu a tam zadat 'tor' enter alebo 'tor | more'
# potom spravca uloh a ci pod CMD procesom bezi tor.exe
## ako SPUSTIS CELE ROTOVANIE TOR IP A STIAHNUT TOR VSTKO TOP VIDEO NAVOD
## https://www.youtube.com/watch?v=wJfa0qEzpJc&t=638s
## tor je v C:\Users\adam\AppData\Roaming\tor\tor


# ------------------------------------------------------------------
# 1) Paths to your Expert Bundle
# ------------------------------------------------------------------
TOR_EXE   = r"C:\Users\Adam\AppData\Roaming\tor\tor\tor.exe"
TORRC     = r"C:\Users\Adam\AppData\Roaming\tor\torrc"

# ------------------------------------------------------------------
# 2) Start Tor if it is not running yet
# ------------------------------------------------------------------
def start_tor():
    # Check if tor.exe is already running
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and proc.info['name'].lower() == 'tor.exe':
            print("Tor is already running.")
            return

    cmd = [TOR_EXE, '-f', TORRC]
    # Launch Tor in a new console window so it survives this script
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    print("Tor started in a new console window.")

    # Wait until SOCKS port 9050 is open
    while True:
        try:
            requests.get('http://ip-api.com/json/',
                         proxies={'http':'socks5h://127.0.0.1:9050'},
                         timeout=1)
            print("SOCKS port 9050 is reachable – Tor is ready.")
            break
        except Exception:
            time.sleep(0.5)

# ------------------------------------------------------------------
# 3) Helpers functions 
# ------------------------------------------------------------------
# Function to get a new Tor identity (rotate IP)
def rotate_tor_identity():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password="fccfui24")  # Replace "your_password" with your Tor control password
        controller.signal(Signal.NEWNYM)

# Make a request through Tor and print the content
def make_tor_request():
    proxy = {'http': 'socks5h://127.0.0.1:9050', 'https': 'socks5h://127.0.0.1:9050'}
    res = requests.get('http://ip-api.com/json/', proxies=proxy)
    print(res.content)

# Check if the website is reachable
def check_website_availability(url):
    try:
        res = requests.get(url)
        res.raise_for_status()  # Raise an HTTPError for bad responses
        print(f"Success! Status code: {res.status_code}")
    except requests.RequestException as e:
        print(f"Failed to reach the website: {e}")

# ------------------------------------------------------------------
# 4) Main flow
# ------------------------------------------------------------------
if __name__ == "__main__":
    start_tor()           # Make sure Tor is running
    rotate_tor_identity() # Rotate identity once
    make_tor_request()   # Do your first request 

# Rotate Tor identity and check the website
#rotate_tor_identity()
#check_website_availability('https://www.reality.sk/')