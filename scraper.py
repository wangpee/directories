#!/usr/bin/env python3
"""
DriveDir AI Price Scraper — powered by Crawl4AI
================================================
Uses Crawl4AI (headless browser) to fetch pages from Nigerian car rental
company websites, then Claude AI to extract vehicle prices intelligently.
Compares against current prices in nigeria-car-rental.html and outputs
a JSON report of detected changes.

Setup:
    pip install crawl4ai anthropic
    crawl4ai-setup          # installs Playwright browsers (one-time)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scraper.py

    # Compare against a specific HTML file:
    python scraper.py --html /path/to/nigeria-car-rental.html

    # Test a single company (faster):
    python scraper.py --company co-nairaxi-luxury-car-rentals

    # Scrape only, no price comparison:
    python scraper.py --dry-run

Output:
    price_changes.json   — structured list of detected price changes
    scrape_log.txt       — full human-readable run log
"""

import os
import re
import json
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Optional

import anthropic

# crawl4ai imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# ─── Configuration ────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Companies with known websites — only those with published pricing pages
COMPANIES = [
    {
        "id": "co-nairaxi-luxury-car-rentals",
        "name": "Nairaxi Luxury Car Rentals",
        "url": "https://nairaxi.ng",
        "wait_for": None,
        "price_hints": ["business class", "vip", "prestige", "per day"],
    },
    {
        "id": "co-naijacarhirecom",
        "name": "NaijaCarhire.com",
        "url": "https://naijacarhire.com/articles/car-collection",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-starr-luxury-cars-abuja",
        "name": "Starr Luxury Cars (Abuja)",
        "url": "https://starrluxurycars.com",
        "wait_for": None,
        "price_hints": ["per day", "daily rate", "hire"],
    },
    {
        "id": "co-jonellies-autos-jautos-car-rentals",
        "name": "Jautos Car Rentals",
        "url": "https://jautos.com.ng",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-sixt-nigeria",
        "name": "SIXT Nigeria",
        "url": "https://www.sixt.com/car-rental/nigeria/",
        "wait_for": "css:.vehicle-list, css:.car-card",
        "price_hints": ["per day", "daily", "from", "price"],
    },
    {
        "id": "co-europcar-nigeria",
        "name": "Europcar Nigeria",
        "url": "https://www.europcar.com/en-gb/car-hire/nigeria",
        "wait_for": "css:.car-item, css:.vehicle-card",
        "price_hints": ["per day", "from", "price", "daily"],
    },
    {
        "id": "co-hertz-nigeria-candi-leasing-plc",
        "name": "Hertz Nigeria",
        "url": "https://www.hertz.com/rentacar/location/nigeria",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate", "from"],
    },
    {
        "id": "co-autopilot-ng",
        "name": "AutoPilot NG",
        "url": "https://autopilot.ng",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate", "hire"],
    },
    {
        "id": "co-aisle-car-rentals",
        "name": "Aisle Car Rentals",
        "url": "https://aisle.com.ng",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-rent-to-drive-nigeria",
        "name": "Rent To Drive Nigeria",
        "url": "https://renttodrive.com.ng",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-lestat-leasing-rent-a-car-nigeria",
        "name": "Lestat Leasing / Rent A Car Nigeria",
        "url": "https://rentacarnigeria.com",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-rideonnigeria",
        "name": "RideOnNigeria",
        "url": "https://rideonnigeria.co",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-fleetpartners-leasing-nigeria",
        "name": "FleetPartners Leasing Nigeria",
        "url": "https://fleetpartners.ng",
        "wait_for": None,
        "price_hints": ["per day", "daily", "lease", "rate"],
    },
    {
        "id": "co-the-lux-auto",
        "name": "The Lux Auto",
        "url": "https://theluxauto.com",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-skymiles-rentals",
        "name": "Skymiles Rentals",
        "url": "https://skymilesrental.com",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-venegow-car-rentals",
        "name": "Venegow Car Rentals",
        "url": "https://venegow.com",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
    {
        "id": "co-magnicarz-auto-sales-and-rentals",
        "name": "Magnicarz Auto Sales & Rentals",
        "url": "https://magnicarz.com",
        "wait_for": None,
        "price_hints": ["per day", "daily", "rate"],
    },
]

MAX_CONTENT_CHARS = 15000

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Crawl4AI fetching ────────────────────────────────────────────────────────

async def fetch_company_page(crawler: AsyncWebCrawler, company: dict) -> Optional[str]:
    """Use Crawl4AI to fetch a company's page. Returns clean markdown or None."""
    url = company["url"]
    wait_for = company.get("wait_for")

    content_filter = PruningContentFilter(threshold=0.4, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    config = CrawlerRunConfig(
        page_timeout=45000,
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "script", "style", "noscript"],
        remove_overlay_elements=True,
        exclude_external_links=True,
        wait_for=wait_for,
    )

    try:
        result = await crawler.arun(url=url, config=config)

        if not result.success:
            log.warning(f"  Crawl4AI error: {result.error_message}")
            return None

        # Prefer noise-filtered fit_markdown
        content = ""
        if hasattr(result.markdown, "fit_markdown") and result.markdown.fit_markdown:
            content = result.markdown.fit_markdown
        elif hasattr(result.markdown, "raw_markdown"):
            content = result.markdown.raw_markdown
        else:
            content = str(result.markdown)

        if not content.strip():
            log.warning(f"  Empty content returned")
            return None

        log.info(f"  Fetched {len(content):,} chars via Crawl4AI")
        return content[:MAX_CONTENT_CHARS]

    except Exception as e:
        log.error(f"  Crawl4AI exception: {e}")
        return None


# ─── Claude price extraction ──────────────────────────────────────────────────

def extract_prices_with_claude(company_name: str, page_content: str) -> list:
    """Send page markdown to Claude and extract structured vehicle prices."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are a data extraction assistant for DriveDir, a Nigerian car rental directory.

Below is markdown scraped from the website of "{company_name}".

Extract ALL vehicle rental prices you can find.

Return ONLY a valid JSON array of objects with these fields:
- "vehicle": vehicle name (e.g. "Toyota Camry", "Mercedes-Benz S-Class 2022")
- "type": one of: "Economy", "Executive", "SUV", "Luxury SUV", "Luxury", "Ultra Luxury", "Bus", "Van", "Supercar"
- "chauffeur_price": daily with-driver price as integer in Naira (null if not stated)
- "self_drive_price": daily self-drive price as integer in Naira (null if not stated)
- "availability": "available", "limited", or "on_request"
- "notes": important notes or null

Rules:
- Price ranges: use the lower bound (₦150,000-200,000 → 150000)
- Strip commas: ₦150,000 → 150000
- "from ₦X": use X
- USD: multiply by 1600. GBP: multiply by 2000
- Single unlabelled price: assume chauffeur rate
- No prices found: return []
- Do NOT invent prices
- Return ONLY raw JSON, no markdown fences

Page content:
---
{page_content}
---"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return []
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return []


# ─── Price comparison ─────────────────────────────────────────────────────────

def extract_current_prices_from_html(html_path: str) -> dict:
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    results = {}
    card_positions = [(m.group(1), m.start())
                      for m in re.finditer(r'data-cid="([^"]+)"', content)]

    for i, (cid, pos) in enumerate(card_positions):
        end = card_positions[i + 1][1] if i + 1 < len(card_positions) else len(content)
        section = content[pos:end]

        vnames = re.findall(r'<div class="vcar-name">([^<]+)</div>', section)
        prices = re.findall(r'<span class="vcar-price">([\d,\u20a6\+]+)', section)

        def parse_price(p):
            if not p:
                return None
            try:
                return int(re.sub(r"[^\d]", "", p))
            except ValueError:
                return None

        vehicles = []
        price_iter = iter(prices)
        for vname in vnames:
            vehicles.append({
                "vehicle": vname.strip(),
                "chauffeur_price": parse_price(next(price_iter, None)),
                "self_drive_price": parse_price(next(price_iter, None)),
            })

        if vehicles:
            results[cid] = vehicles

    return results


def word_overlap(a: str, b: str) -> int:
    return len(set(a.lower().split()) & set(b.lower().split()))


def compare_prices(current: list, scraped: list) -> list:
    changes = []
    changes.append({"vehicle": "TEST VEHICLE", "note": "This is a test alert — delete me"})
    return changes
    for sv in scraped:
        sv_name = sv.get("vehicle", "")
        best = max(current, key=lambda c: word_overlap(sv_name, c.get("vehicle", "")), default=None)

        if best and word_overlap(sv_name, best.get("vehicle", "")) >= 1:
            diffs = {}
            for field in ("chauffeur_price", "self_drive_price"):
                s_val, c_val = sv.get(field), best.get(field)
                if s_val and c_val and s_val != c_val:
                    pct = abs(s_val - c_val) / c_val * 100
                    if pct > 0:
                        diffs[field] = {"old": c_val, "new": s_val, "change_pct": round(pct, 1)}
            if diffs:
                changes.append({"vehicle": sv_name, "matched_to": best["vehicle"], "changes": diffs})
        elif sv.get("chauffeur_price") or sv.get("self_drive_price"):
            changes.append({
                "vehicle": sv_name, "matched_to": None,
                "note": "NEW vehicle found — not in current DriveDir listing",
                "chauffeur_price": sv.get("chauffeur_price"),
                "self_drive_price": sv.get("self_drive_price"),
                "type": sv.get("type"),
            })
    return changes


# ─── Main ─────────────────────────────────────────────────────────────────────

async def run(args):
    if not ANTHROPIC_API_KEY:
        log.error("Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY=sk-ant-...")
        return

    log.info("=" * 65)
    log.info(f"DriveDir Price Scraper (Crawl4AI)  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 65)

    current_prices = {}
    if not args.dry_run:
        if os.path.exists(args.html):
            current_prices = extract_current_prices_from_html(args.html)
            log.info(f"Loaded current prices for {len(current_prices)} companies from {args.html}\n")
        else:
            log.warning(f"HTML file not found — running dry-run\n")
            args.dry_run = True

    companies = COMPANIES
    if args.company:
        companies = [c for c in COMPANIES if c["id"] == args.company]
        if not companies:
            log.error(f"Unknown company ID: {args.company}")
            return

    all_results = []

    browser_config = BrowserConfig(
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, company in enumerate(companies):
            cid, name, url = company["id"], company["name"], company["url"]
            log.info(f"[{i+1}/{len(companies)}] {name}")
            log.info(f"  URL: {url}")

            page_content = await fetch_company_page(crawler, company)

            if not page_content:
                log.warning("  Skipping — fetch failed\n")
                all_results.append({
                    "company_id": cid, "company_name": name, "url": url,
                    "status": "fetch_failed", "scraped_at": datetime.now().isoformat(),
                    "scraped_vehicles": [], "changes": [],
                })
                await asyncio.sleep(2)
                continue

            # Check for price keywords
            if not any(h in page_content.lower() for h in company.get("price_hints", [])):
                log.info("  No price keywords found — prices may not be published on this page")

            log.info("  Extracting prices with Claude AI...")
            scraped = extract_prices_with_claude(name, page_content)

            if scraped:
                log.info(f"  Found {len(scraped)} vehicle(s):")
                for v in scraped:
                    cp = f"\u20a6{v['chauffeur_price']:,}" if v.get("chauffeur_price") else "—"
                    sp = f"\u20a6{v['self_drive_price']:,}" if v.get("self_drive_price") else "—"
                    log.info(f"     {v.get('vehicle', '?'):<35} chauffeur={cp}  self-drive={sp}")
            else:
                log.info("  No prices extracted")

            changes = []
            if not args.dry_run and cid in current_prices and scraped:
                changes = compare_prices(current_prices[cid], scraped)
                if changes:
                    log.info(f"  CHANGES DETECTED ({len(changes)}):")
                    for ch in changes:
                        log.info(f"    {ch['vehicle']}")
                        for field, vals in ch.get("changes", {}).items():
                            arrow = "up" if vals["new"] > vals["old"] else "down"
                            log.info(f"      {field}: {vals['old']:,} -> {vals['new']:,} ({arrow} {vals['change_pct']:.1f}%)")
                        if "note" in ch:
                            log.info(f"      {ch['note']}")
                else:
                    log.info("  No significant changes")

            all_results.append({
                "company_id": cid, "company_name": name, "url": url,
                "status": "success" if scraped else "no_prices_found",
                "scraped_at": datetime.now().isoformat(),
                "scraped_vehicles": scraped,
                "changes": changes,
            })
            log.info("")
            await asyncio.sleep(2)

    # Summary
    total_v = sum(len(r["scraped_vehicles"]) for r in all_results)
    total_c = sum(len(r["changes"]) for r in all_results)
    n_ok    = sum(1 for r in all_results if r["status"] == "success")
    n_np    = sum(1 for r in all_results if r["status"] == "no_prices_found")
    n_fail  = sum(1 for r in all_results if r["status"] == "fetch_failed")

    log.info("=" * 65)
    log.info(f"SUMMARY: {len(all_results)} companies | {n_ok} with prices | {n_fail} failed | {total_v} vehicles | {total_c} changes")
    log.info("=" * 65)

    if total_c > 0:
        log.info("CHANGES REQUIRING REVIEW:")
        for r in all_results:
            if r["changes"]:
                log.info(f"  {r['company_name']}")
                for ch in r["changes"]:
                    log.info(f"    - {ch['vehicle']}: {ch.get('changes', ch.get('note', ''))}")

    output = {
        "generated_at": datetime.now().isoformat(),
        "html_file": args.html,
        "summary": {
            "companies_scraped": len(all_results),
            "prices_found": n_ok, "no_prices_found": n_np, "fetch_failed": n_fail,
            "total_vehicles_found": total_v, "total_changes_detected": total_c,
        },
        "results": all_results,
    }

    with open("price_changes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("Results saved to price_changes.json")
    log.info("Log saved to scrape_log.txt")
    if total_c > 0:
        log.info("ACTION NEEDED: review price_changes.json and update nigeria-car-rental.html")


def main():
    parser = argparse.ArgumentParser(description="DriveDir Price Scraper (Crawl4AI)")
    parser.add_argument("--html", default="nigeria-car-rental.html")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--company", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
