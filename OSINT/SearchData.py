#!apt-get update
#!apt-get install -y chromium-chromedriver
#!pip install selenium webdriver-manager pillow
#!pip install selenium webdriver-manager pillow undetected-chromedriver
#!pip install -U yt-dlp

#!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && apt install ./google-chrome-stable_current_amd64.deb

import os
import time
#from google.colab import drive
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess
from datetime import date

# Logging einrichten
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_driver():
    """Initialisiert den Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

def download_videos(username, max_videos=10):
    """Lädt Videos eines Twitter-Users mit yt-dlp in Google Drive"""
    try:
        # Google Drive verbinden
        #drive.mount('/content/drive', force_remount=True)

        # Speicherort setzen
        heutiges_datum = date.today().strftime("%Y-%m-%d")
        save_path = f"/home/user/OSINTxDeepfakeBench/OSINT/Research_Data/{username}_{heutiges_datum}"
        os.makedirs(save_path, exist_ok=True)

        # yt-dlp installieren
        #subprocess.run(["pip", "install", "yt-dlp"], check=True)

        # WebDriver starten
        driver = setup_driver()
        url = f"https://x.com/{username}"

        logger.info(f"Öffne Twitter-Profil: {url}")
        driver.get(url)

        # Warte auf das Laden der ersten Tweets
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )
        logger.info("Erste Tweets geladen.")

        # Dynamisches Scrollen mit optimierten Wartezeiten
        max_scroll_attempts = 20
        scroll_attempt = 0
        previous_tweet_count = 0

        while scroll_attempt < max_scroll_attempts:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            try:
                # Warte bis zu 5 Sekunden, bis neue Tweets geladen sind
                WebDriverWait(driver, 5).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, "article")) > previous_tweet_count
                )
            except:
                logger.info("Keine neuen Tweets mehr geladen oder Timeout erreicht.")
                break

            current_tweets = driver.find_elements(By.CSS_SELECTOR, "article")
            current_tweet_count = len(current_tweets)
            if current_tweet_count == previous_tweet_count:
                logger.info("Keine neuen Tweets mehr geladen.")
                break
            previous_tweet_count = current_tweet_count
            scroll_attempt += 1
            logger.info(f"Scroll-Versuch {scroll_attempt}, Tweets: {current_tweet_count}")

        # Tweet-URLs in der Reihenfolge sammeln und Duplikate entfernen
        tweet_urls = []
        seen_urls = set()
        tweets = driver.find_elements(By.CSS_SELECTOR, "article")
        for tweet in tweets:
            try:
                link_element = tweet.find_element(By.CSS_SELECTOR, "a[href*='/status/']")
                tweet_url = link_element.get_attribute("href")
                if tweet_url not in seen_urls:
                    tweet_urls.append(tweet_url)
                    seen_urls.add(tweet_url)
            except:
                continue

        logger.info(f"Gefundene einzigartige Tweets: {len(tweet_urls)}")

        # Videos mit yt-dlp herunterladen
        downloaded = 0
        for tweet_url in tweet_urls:
            if downloaded >= max_videos:
                break
            try:
                output_template = os.path.join(save_path, f"{username}_video_%(id)s.%(ext)s")
                result = subprocess.run([
                    "yt-dlp",
                    "--output", output_template,
                    "--no-playlist",
                    "--quiet",
                    "--no-warnings",
                    tweet_url
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    downloaded += 1
                    logger.info(f"Gespeichert: Video von {tweet_url}")
                else:
                    logger.debug(f"Kein Video in {tweet_url}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Fehler bei {tweet_url}: {e}")
                continue

        return downloaded

    except Exception as e:
        logger.error(f"Ein Fehler ist aufgetreten: {e}")
        return 0

    finally:
        try:
            driver.quit()
        except:
            pass

# Beispielaufruf
if __name__ == "__main__":
    target_username = input("Bitte gib den Twitter-Username ein: ")
    downloaded = download_videos(target_username, max_videos=10)
    logger.info(f"Insgesamt {downloaded} Videos heruntergeladen")
