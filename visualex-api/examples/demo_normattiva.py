
import asyncio
import logging
import sys
from visualex.scrapers.normattiva import NormattivaScraper
from visualex.models.norma import NormaVisitata, Norma

logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing Normattiva Scraper Demo...")
    scraper = NormattivaScraper()
    
    # Example URN: Costituzione (urn:nir:stato:costituzione:1947-12-27;nir-1)
    norma = Norma(tipo_atto="costituzione", data="1947-12-27", numero_atto="nir-1")
    norma_visitata = NormaVisitata(norma=norma)
    
    print(f"Fetching URN: {norma_visitata.urn}")
    try:
        content, fetched_urn = await scraper.get_document(norma_visitata)
        print("--- Document Content Start ---")
        print(content[:500] + "...")
        print("--- Document Content End ---")
        print(f"Fetched URN: {fetched_urn}")
    except Exception as e:
        print(f"Error fetching document: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
