import requests

urls = [
    "https://images.unsplash.com/photo-1510206109315-99d86b7617b0?q=80&w=1000&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1599839619722-39751411ea63?q=80&w=1000&auto=format&fit=crop"
]

for i, url in enumerate(urls):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        filepath = rf'c:\Users\ASUS\OneDrive\Desktop\FYP 2.0\data\inputs\traffic_sample_{i+1}.jpg'
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filepath}")
    else:
        print(f"Failed to download {url}")
