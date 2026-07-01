import requests
from bs4 import BeautifulSoup

url = "https://example-college-review-site.com/vit"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

reviews = soup.find_all("div", class_="review-text")

data = []
for r in reviews:
    data.append({
        "college": "VIT",
        "review_text": r.text.strip(),
        "source": "college_review_site"
    })