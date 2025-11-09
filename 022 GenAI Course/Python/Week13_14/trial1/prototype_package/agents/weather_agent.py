import os, requests, time
class WeatherAgent:
    def __init__(self, api_key_env='OPENWEATHER_API_KEY'):
        self.api_key = os.getenv(api_key_env)

    def get_weather(self, location):
        if not self.api_key:
            # Mocked fallback (deterministic)
            return f"Mock weather for {location}: 22°C, clear skies. (Set OPENWEATHER_API_KEY to enable real data)"
        try:
            # call OpenWeatherMap current weather API (city name)
            url = 'https://api.openweathermap.org/data/2.5/weather'
            params = {'q': location, 'appid': self.api_key, 'units':'metric'}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"Weather in {location}: {temp}°C, {desc}."
        except Exception as e:
            return f"Weather API error: {e}"
