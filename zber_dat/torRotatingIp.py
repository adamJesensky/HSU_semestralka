import requests

#https://www.youtube.com/watch?v=Ep5ytx8ad04
#treba ist do TOR zlozky C:\programs\Tor Browser\Browser\TorBrowser\Tor, pustit hore cmd cez adresu a tam zadat 'tor' enter alebo 'tor | more'
# potom spravca uloh a ci pod CMD procesom bezi tor.exe


## ako SPUSTIS CELE ROTOVANIE TOR IP A STAIHNUT TOR VSTKO TOP VIDEO
## https://www.youtube.com/watch?v=wJfa0qEzpJc&t=638s
## tor je v C:\Users\adam\AppData\Roaming\tor\tor

from stem import Signal
from stem.control import Controller

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

# Rotate Tor identity and make another request
rotate_tor_identity()
make_tor_request()

# Rotate Tor identity and check the website
#rotate_tor_identity()
#check_website_availability('https://www.reality.sk/')
