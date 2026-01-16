import requests

API_KEY = "71810bc9e6616173aeb1be679d4b87b9"

def get_weather(city):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        return {"error": data.get("message", "Weather data unavailable")}

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
        "description": data["weather"][0]["description"].title()
    }
