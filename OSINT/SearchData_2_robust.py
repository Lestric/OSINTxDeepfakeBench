import os
import shutil
import logging
import subprocess
from datetime import date

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Logging einrichten
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# WebDriver-Setup (Headless)
# ------------------------------
def setup_driver():
    """Initialisiert den Chrome WebDriver (headless)."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=de-DE,de")
    options.add_argument("user-agent=Mozilla/5.0 ... Chrome/126 Safari/537.36")

    # Falls erforderlich, Binary explizit setzen:
    # options.binary_location = "/usr/bin/google-chrome"  # oder "/usr/bin/chromium-browser"
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ------------------------------
# Scroll-Helper
# ------------------------------
def scroll_and_collect_articles(driver, max_scroll_attempts=20, wait_secs=5):
    """Scrollt die Profilseite und gibt alle gefundenen <article>-Elemente zurück."""
    previous_count = 0
    for attempt in range(1, max_scroll_attempts + 1):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            WebDriverWait(driver, wait_secs).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "article")) > previous_count
            )
        except Exception:
            logger.info("Keine neuen Tweets mehr geladen oder Timeout erreicht.")
            break
        current = driver.find_elements(By.CSS_SELECTOR, "article")
        if len(current) == previous_count:
            logger.info("Keine Zunahme der Tweets  Abbruch des Scrollens.")
            break
        previous_count = len(current)
        logger.info(f"Scroll {attempt}: Tweets insgesamt {previous_count}")
    return driver.find_elements(By.CSS_SELECTOR, "article")

# ------------------------------
# Strategien AD
# ------------------------------
def extract_status_urls_with_strategies(driver):
    """
    Liefert eine (geordnete) Liste an Status-URLs, die nach vier Strategien gesammelt wird:
      A) Es existiert ein <video>-Tag im Artikel.
      B) Link-Extraktion + lokaler Video-Nachweis im selben Artikel.
      C) Alternative Selektoren (aria-label enthält 'Video', XPath auf //video).
      D) Fallback: Alle status-Links (yt-dlp prüft selbst).
    Reihenfolge wird beibehalten; Duplikate werden vermieden.
    """
    seen = set()
    ordered_urls = []

    def add_url(href: str):
        if href and "/status/" in href and href not in seen:
            seen.add(href)
            ordered_urls.append(href)

    articles = scroll_and_collect_articles(driver)

    # Gemeinsamer Helper: status-Link im Artikel finden
    def find_status_link(elem):
        try:
            link = elem.find_element(By.CSS_SELECTOR, "a[href*='/status/']")
            return link.get_attribute("href")
        except Exception:
            return None

    # ---------------- A) <video>-Tag im Artikel ----------------
    for art in articles:
        try:
            videos = art.find_elements(By.TAG_NAME, "video")
            if videos:
                url = find_status_link(art)
                add_url(url)
        except Exception:
            continue

    # ---------------- B) Link + Video-Nachweis im selben Artikel ----------------
    # (strengere Variante als A, faktisch schon durch A abgedeckt  hier exemplarisch via CSS/XPath)
    for art in articles:
        try:
            url = find_status_link(art)
            if not url:
                continue
            has_html5_video = len(art.find_elements(By.TAG_NAME, "video")) > 0
            has_possible_inline = len(art.find_elements(By.CSS_SELECTOR, "div[role='presentation']")) > 0
            if has_html5_video or has_possible_inline:
                add_url(url)
        except Exception:
            continue

    # ---------------- C) Alternative Selektoren ----------------
    # C1: aria-label mit 'Video'
    for art in articles:
        try:
            # X/Twitter nutzt dynamische Strukturen; aria-label-Hinweise sind relativ stabil
            hint = art.find_elements(By.CSS_SELECTOR, "[aria-label*='Video'], [aria-label*='video']")
            if hint:
                url = find_status_link(art)
                add_url(url)
        except Exception:
            continue

    # C2: XPath-Fallback auf <video>-Tags (robust gegen verschachtelte Strukturen)
    for art in articles:
        try:
            vids = art.find_elements(By.XPATH, ".//video")
            if vids:
                url = find_status_link(art)
                add_url(url)
        except Exception:
            continue

    # ---------------- D) Fallback: Alle status-Links (yt-dlp filtert) ----------------
    # Ergänzt ggf. übersehene Tweets; yt-dlp erkennt selbst, ob ein Video vorhanden ist.
    try:
        all_status_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/status/']")
        for a in all_status_links:
            add_url(a.get_attribute("href"))
    except Exception:
        pass

    logger.info(f"Gesamtzahl Kandidaten-URLs (AD, unique): {len(ordered_urls)}")
    return ordered_urls

# ------------------------------
# Download mit yt-dlp
# ------------------------------
def run_ytdlp(url, out_template, cookies_from_browser=None, quiet=True):
    """Führt yt-dlp aus; gibt True zurück, wenn ein Download stattfand (rc==0)."""
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp nicht gefunden. Bitte mit 'pip install yt-dlp' installieren.")

    args = ["yt-dlp", "--no-playlist", "--no-warnings", "--output", out_template]
    if quiet:
        args.append("--quiet")
    if cookies_from_browser:  # z. B. "chrome" oder "chromium"
        args += ["--cookies-from-browser", cookies_from_browser]
    args.append(url)

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        # Debug-Ausgabe bei Bedarf:
        logger.debug(f"yt-dlp rc={result.returncode} url={url} stderr={result.stderr.strip()}")
        return False
    return True

# ------------------------------
# Hauptfunktion
# ------------------------------
def download_videos(username, max_videos=10, cookies_from_browser=None):
    """Identifiziert Video-Tweets (AD) und lädt sie mit yt-dlp herunter."""
    driver = None
    try:
        # Zielverzeichnis
        heutiges_datum = date.today().strftime("%Y-%m-%d")
        base_dir = os.path.join(os.path.expanduser("~"), "OSINTxDeepfakeBench/OSINT/Research_Data")
        save_path = os.path.join(base_dir, f"{username}_{heutiges_datum}")
        os.makedirs(save_path, exist_ok=True)

        # Seite öffnen
        driver = setup_driver()
        url = f"https://x.com/{username}/media"
        logger.info(f"Öffne Profil: {url}")
        driver.get(url)
        


        # Erste Artikel abwarten
        WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))
        logger.info("Erste Tweets geladen.")

        # Kandidaten-URLs via AD extrahieren
        candidate_urls = extract_status_urls_with_strategies(driver)

        # Downloads starten
        downloaded = 0
        for tweet_url in candidate_urls:
            if downloaded >= max_videos:
                break
            out_tmpl = os.path.join(save_path, f"{username}_video_%(id)s.%(ext)s")
            ok = run_ytdlp(tweet_url, out_tmpl, cookies_from_browser=cookies_from_browser, quiet=True)
            if ok:
                downloaded += 1
                logger.info(f"Gespeichert: {tweet_url}")
            else:
                logger.debug(f"Kein Video gefunden oder Download fehlgeschlagen: {tweet_url}")

        logger.info(f"Insgesamt {downloaded} Videos heruntergeladen.")
        return downloaded

    except Exception as e:
        logger.error(f"Fehler: {e}")
        return 0
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    target_username = input("Bitte gib den Twitter-Username ein: ").strip()
    # Falls Sie eingeloggt sind und X-Inhalte sonst beschränkt sind, aktivieren Sie Cookies:
    # downloaded = download_videos(target_username, max_videos=10, cookies_from_browser="chrome")
    downloaded = download_videos(target_username, max_videos=10, cookies_from_browser=None)
    logger.info(f"Fertig. Anzahl gespeicherter Videos: {downloaded}")

