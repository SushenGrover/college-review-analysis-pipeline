import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import time
import random

# --- CONFIGURATION ---
BASE_URL = "https://www.careers360.com/colleges/reviews?page={}"
MAX_WORKERS = 5  # Number of simultaneous downloads (Don't go too high!)
START_PAGE = 1
ESTIMATED_LAST_PAGE = 5438 # Adjust this based on how many reviews you want (e.g., 5000)
OUTPUT_FILENAME = "careers360_raw_reviews.csv"

# Shared session for speed (Reuses TCP connections)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/"
})

def clean_text(text):
    if text:
        return " ".join(text.split())
    return "N/A"

def fetch_page(page):
    """Fetches and extracts reviews from a single page."""
    url = BASE_URL.format(page)
    extracted_reviews = []
    
    try:
        # Small random delay to reduce bot detection risk
        time.sleep(random.uniform(0.5, 1.5))
        
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️ Page {page} failed with status {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        cards = soup.find_all("div", class_="collegeReviewlisting_card")

        if not cards:
            return [] # Empty list indicates no reviews found

        for card in cards:
            # 1. College Name
            name_tag = card.find("a", class_="college_name")
            college_name = clean_text(name_tag.text) if name_tag else "N/A"

            # 2. Review Title
            title_tag = card.find("h2", class_="headingH2")
            review_title = clean_text(title_tag.text) if title_tag else "N/A"

            # 3. Overall Rating
            rating_tag = card.find("div", class_="star-ratings")
            if rating_tag and rating_tag.has_attr("title"):
                overall_rating = rating_tag["title"].split()[0]
            else:
                overall_rating = "N/A"

            # 4. Posted Info
            author_div = card.find("div", class_="author_content")
            posted_info = clean_text(author_div.text) if author_div else "N/A"

            # 5. Full Review Text (Merging sections)
            review_parts = []
            detail_blocks = card.find_all("div", class_="detail_content_review")
            for block in detail_blocks:
                header = block.find("h4")
                content = block.find("p")
                if header and content:
                    review_parts.append(f"[{clean_text(header.text)}]: {clean_text(content.text)}")
            
            full_review_text = " | ".join(review_parts) if review_parts else "N/A"

            extracted_reviews.append({
                "college_name": college_name,
                "review_title": review_title,
                "overall_rating": overall_rating,
                "review_text": full_review_text,
                "posted_info": posted_info,
                "source": "careers360",
                "page_number": page
            })
            
        print(f"✅ Page {page} scraped ({len(extracted_reviews)} reviews)")
        return extracted_reviews

    except Exception as e:
        print(f"❌ Error on page {page}: {e}")
        return []

def main():
    print(f"🚀 Starting fast scraper with {MAX_WORKERS} workers...")
    
    all_data = []
    
    # Create a list of pages to scrape
    # Note: Since we don't know the exact end, we set a high range.
    # The code handles empty pages gracefully.
    pages_to_scrape = range(START_PAGE, ESTIMATED_LAST_PAGE + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        future_to_page = {executor.submit(fetch_page, page): page for page in pages_to_scrape}
        
        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                data = future.result()
                if data:
                    all_data.extend(data)
                else:
                    print(f"⚠️ No data on page {page_num} (might be end of list or empty).")
            except Exception as exc:
                print(f"Page {page_num} generated an exception: {exc}")

    # --- SAVE TO CSV ---
    if all_data:
        # Sort by page number just to keep it tidy
        all_data.sort(key=lambda x: x['page_number'])
        
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print(f"\n🎉 Finished! Scraped {len(df)} total reviews.")
        print(f"📂 Saved to: {OUTPUT_FILENAME}")
    else:
        print("\n❌ No reviews scraped.")

if __name__ == "__main__":
    main()