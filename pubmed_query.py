import asyncio
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm import tqdm

API_KEY = insert_api_key_here

API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

def load_dois():
    """
    Reads the CSV file and returns a list of DOIs.
    The CSV should have a column named 'doi'.
    """
    df_biomedrxiv = pd.read_csv('filtered_biomedrxiv_data.csv')
    dois = df_biomedrxiv["doi"].tolist()
    return dois

async def check_doi_in_pubmed(session, doi, limiter):
    """
    Asynchronously queries the PubMed API for a given DOI.
    Implements retry logic and respects the rate limiter.
    """
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]",
        "retmode": "json",
        "api_key": API_KEY
    }
    retries = 3
    while retries > 0:
        async with limiter:
            try:
                async with session.get(API_URL, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        pmids = data.get("esearchresult", {}).get("idlist", [])
                        return {"DOI": doi, "PMID": pmids[0] if pmids else None}
                    elif response.status == 429:
                        print(f"Rate limit hit for DOI {doi}, sleeping for 2 seconds...")
                        await asyncio.sleep(2)
                    else:
                        print(f"Error {response.status} for DOI: {doi}")
                        return {"DOI": doi, "PMID": None}
            except Exception as e:
                print(f"Request error for DOI {doi}: {e}")
        retries -= 1
        await asyncio.sleep(2)
    return {"DOI": doi, "PMID": None}

async def main():
    dois = load_dois()
    print(f"Found {len(dois)} DOIs to process.")
    
    limiter = AsyncLimiter(max_rate=3, time_period=1)
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [check_doi_in_pubmed(session, doi, limiter) for doi in dois]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing DOIs"):
            result = await future
            results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv("pubmed_results.csv", index=False)
    print("Results saved in 'pubmed_results.csv'.")

if __name__ == "__main__":
    asyncio.run(main())