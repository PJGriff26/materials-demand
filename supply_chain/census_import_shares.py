"""Fetch US import shares by source country from the Census Bureau
International Trade API.

Optionally refreshes the cached US import-partner shares behind the
supply-chain risk metrics; results are written to a local JSON cache so the
rest of the pipeline runs offline. Per material, queries annual general-import
value by partner country, aggregates across HTS codes, and returns a
[material, country, share] DataFrame.
"""

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────────

CENSUS_API_BASE = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

# Cache file to avoid repeated API calls during development / reruns
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "census_trade"
CACHE_FILE = CACHE_DIR / "import_shares_cache.json"

# US Census country code for "World" total (used for validation, not queried)
# We query all partners and compute shares from the results.

# Average over this many years to smooth annual fluctuations,
# matching USGS methodology (MCS reports "2020-2023" averages).
DEFAULT_YEARS = [2020, 2021, 2022, 2023]

# ── HTS Code Mapping ────────────────────────────────────────────────────────
# Each material maps to a list of HTS codes (10-digit, no dots).
# Multiple codes per material are summed before computing shares.
# Codes selected to match the import forms used in USGS MCS reporting.

HTS_CODES = {
    # --- Core metals (19) ---
    # NOTE: Some codes are HS6 (6 digits) because the Census API does not
    # serve data at HS10 for all commodities. The query functions handle both.
    "Aluminum": [
        "760110",  # Unwrought aluminum, not alloyed (primary) [HS6]
        "760120",  # Unwrought aluminum alloys [HS6]
    ],
    "Boron": [
        "252800",  # Natural borates and concentrates [HS6]
        "2810000000",  # Oxides of boron; boric acids
    ],
    "Cement": [
        "2523290000",  # Portland cement (other than white)
        "2523100000",  # Cement clinkers
    ],
    "Chromium": [
        "261000",      # Chromium ores and concentrates [HS6]
        "7202410000",  # Ferrochromium, >4% C
        "720249",      # Ferrochromium, ≤4% C [HS6]
    ],
    "Copper": [
        "7403110000",  # Refined copper cathodes
        "7403190000",  # Other unwrought refined copper
    ],
    "Lead": [
        "7801100000",  # Refined lead, unwrought
    ],
    "Magnesium": [
        "8104110000",  # Unwrought Mg, ≥99.8%
        "8104190000",  # Unwrought Mg, <99.8%
    ],
    "Manganese": [
        "260200",      # Manganese ores and concentrates [HS6]
        "720211",      # Ferromanganese, >2% C [HS6]
        "720219",      # Ferromanganese, ≤2% C [HS6]
    ],
    "Molybdenum": [
        "2613100000",  # Molybdenum ores, roasted
        "7202700000",  # Ferromolybdenum
        "2825700000",  # Molybdenum oxides and hydroxides
    ],
    "Nickel": [
        "7502100000",  # Nickel, unwrought, not alloyed
    ],
    "Niobium": [
        "720293",      # Ferroniobium (ferrocolumbium) [HS6]
    ],
    "Rare Earths": [
        "284610",      # Cerium compounds [HS6]
        "284690",      # Other rare-earth compounds [HS6]
        "280530",      # Rare-earth metals, Sc, Y [HS6]
    ],
    "Silicon": [
        "280469",      # Silicon, <99.99% (metallurgical grade) [HS6]
        "2804610000",  # Silicon, ≥99.99%
        "720221",      # Ferrosilicon, >55% Si [HS6]
        "720229",      # Ferrosilicon, ≤55% Si [HS6]
    ],
    "Silver": [
        "710691",      # Silver bullion, unwrought [HS6]
    ],
    "Steel": [
        "7201100000",  # Nonalloy pig iron
        "7207110000",  # Semi-finished steel, billets
        "7207120000",  # Semi-finished steel, slabs
    ],
    "Tin": [
        "8001100000",  # Unwrought tin, not alloyed
    ],
    "Vanadium": [
        "7202920000",  # Ferrovanadium
        "2825300010",  # Vanadium pentoxide
    ],
    "Yttrium": [
        "2846902015",  # Yttrium compounds
    ],
    "Zinc": [
        "7901110000",  # Unwrought zinc, ≥99.99%
        "7901120000",  # Unwrought zinc, <99.99%
    ],
    # --- Thin-film / byproduct metals (6) ---
    "Cadmium": [
        "8107200000",  # Unwrought cadmium; powders
    ],
    "Gallium": [
        "8112921000",  # Unwrought gallium
    ],
    "Germanium": [
        "8112926000",  # Unwrought germanium
    ],
    "Indium": [
        "8112923000",  # Unwrought indium; powders
    ],
    "Selenium": [
        "2804900000",  # Selenium
    ],
    "Tellurium": [
        "2804500020",  # Tellurium
    ],
}

# Materials in our pipeline
ALL_MATERIALS = list(HTS_CODES.keys())


# ── API Query ────────────────────────────────────────────────────────────────

def _query_census_api(
    hts_code: str,
    years: list[int],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query Census Bureau International Trade API for import values by country.

    Parameters
    ----------
    hts_code : str
        10-digit HTS code (no dots).
    years : list of int
        Calendar years to query.
    api_key : str, optional
        Census API key. Without one, limited to 500 calls/day.

    Returns
    -------
    pd.DataFrame
        Columns: [country, country_code, value, year]
        where value is general import value in USD.
    """
    # Determine comm level from code length
    comm_lvl = "HS6" if len(hts_code) == 6 else "HS10"

    rows = []
    for year in years:
        params = {
            "get": "CTY_NAME,CTY_CODE,GEN_VAL_YR",
            "COMM_LVL": comm_lvl,
            "I_COMMODITY": hts_code,
            "time": str(year),
        }
        if api_key:
            params["key"] = api_key

        try:
            resp = requests.get(CENSUS_API_BASE, params=params, timeout=30)
            if resp.status_code == 204 or len(resp.content) == 0:
                # No data at HS10 level — will try HS6 fallback after loop
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"    WARNING: Census API error for {hts_code}/{year}: {e}")
            continue
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"    WARNING: Census API request failed for {hts_code}/{year}: {e}")
            continue

        if not data or len(data) < 2:
            continue

        header = data[0]
        for row in data[1:]:
            record = dict(zip(header, row))
            val = record.get("GEN_VAL_YR", "0")
            if val in (None, "", "null", "-"):
                val = 0
            rows.append({
                "country": record.get("CTY_NAME", "Unknown"),
                "country_code": record.get("CTY_CODE", ""),
                "value": int(val),
                "year": year,
            })

        # Respect rate limits (500/day without key)
        time.sleep(0.5)

    result = pd.DataFrame(rows)

    # If HS10 returned nothing, try HS6 fallback
    if result.empty and comm_lvl == "HS10":
        hs6 = hts_code[:6]
        return _query_census_api_hs6(hs6, years, api_key)

    return result


def _query_census_api_hs6(
    hs6_code: str,
    years: list[int],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fallback: query at HS 6-digit level if 10-digit not available."""
    rows = []
    for year in years:
        params = {
            "get": "CTY_NAME,CTY_CODE,GEN_VAL_YR",
            "COMM_LVL": "HS6",
            "I_COMMODITY": hs6_code,
            "time": str(year),
        }
        if api_key:
            params["key"] = api_key

        try:
            resp = requests.get(CENSUS_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    WARNING: Census API HS6 fallback failed for {hs6_code}/{year}: {e}")
            continue

        if not data or len(data) < 2:
            continue

        header = data[0]
        for row in data[1:]:
            record = dict(zip(header, row))
            val = record.get("GEN_VAL_YR", "0")
            if val in (None, "", "null", "-"):
                val = 0
            rows.append({
                "country": record.get("CTY_NAME", "Unknown"),
                "country_code": record.get("CTY_CODE", ""),
                "value": int(val),
                "year": year,
            })

        time.sleep(0.5)

    return pd.DataFrame(rows)


# ── Country Name Normalization ───────────────────────────────────────────────

# Census API country names → names matching CRC / pipeline conventions
COUNTRY_RENAMES = {
    "CANADA": "Canada",
    "CHINA": "China",
    "MEXICO": "Mexico",
    "JAPAN": "Japan",
    "GERMANY": "Germany",
    "BRAZIL": "Brazil",
    "AUSTRALIA": "Australia",
    "INDIA": "India",
    "RUSSIA": "Russia",
    "SOUTH AFRICA": "South Africa",
    "CHILE": "Chile",
    "PERU": "Peru",
    "SOUTH KOREA": "Republic of Korea",
    "KOREA, SOUTH": "Republic of Korea",
    "UNITED KINGDOM": "United Kingdom",
    "FRANCE": "France",
    "BELGIUM": "Belgium",
    "NETHERLANDS": "Netherlands",
    "NORWAY": "Norway",
    "FINLAND": "Finland",
    "TURKEY": "Turkey",
    "INDONESIA": "Indonesia",
    "BOLIVIA": "Bolivia",
    "PHILIPPINES": "Philippines",
    "KAZAKHSTAN": "Kazakhstan",
    "GABON": "Gabon",
    "MALAYSIA": "Malaysia",
    "ISRAEL": "Israel",
    "AUSTRIA": "Austria",
    "POLAND": "Poland",
    "UNITED ARAB EMIRATES": "United Arab Emirates",
    "BAHRAIN": "Bahrain",
    "JAMAICA": "Jamaica",
    "GUYANA": "Guyana",
    "TAIWAN": "Taiwan",
    "THAILAND": "Thailand",
    "SINGAPORE": "Singapore",
    "ARGENTINA": "Argentina",
    "CZECH REPUBLIC": "Czechia",
    "CZECHIA": "Czechia",
    "ITALY": "Italy",
    "SPAIN": "Spain",
    "VIETNAM": "Vietnam",
    "MOZAMBIQUE": "Mozambique",
    "MADAGASCAR": "Madagascar",
    "ZIMBABWE": "Zimbabwe",
    "SENEGAL": "Senegal",
    "ESTONIA": "Estonia",
    "CONGO (KINSHASA)": "Democratic Republic of the Congo",
    "GREECE": "Greece",
    "COLOMBIA": "Colombia",
    "DOMINICAN REPUBLIC": "Dominican Republic",
    "HONG KONG": "China",  # USGS convention: HK included with China
    "SWEDEN": "Sweden",
    "SWITZERLAND": "Switzerland",
    "IRELAND": "Ireland",
    "EGYPT": "Egypt",
    "JORDAN": "Jordan",
    "MOROCCO": "Morocco",
    "SAUDI ARABIA": "Saudi Arabia",
}


def _normalize_country(name: str) -> str:
    """Normalize Census API country name to pipeline convention."""
    upper = name.strip().upper()
    if upper in COUNTRY_RENAMES:
        return COUNTRY_RENAMES[upper]
    # Title-case fallback for unrecognized names
    return name.strip().title()


# ── Excluded country codes ───────────────────────────────────────────────────
# Census uses special codes for aggregates and trade blocs.
# Real countries have 4-digit numeric codes (e.g., 1220=Canada, 5700=China).
# Aggregates use: "-" (world total), "00XX" (trade blocs like EU, OECD, NATO,
# APEC, LAFTA), or "XXXX" patterns with X (continent subtotals).

def _is_aggregate_code(code: str) -> bool:
    """Return True if this Census country code is an aggregate, not a country."""
    code = str(code).strip()
    if code in ("-", ""):
        return True
    # "0003"=EU, "0014"=Pacific Rim, "0020"=NAFTA, "0021"=LATAM,
    # "0022"=OECD, "0023"=NATO, "0024"=LAFTA, "0025"=Euro Area, "0026"=APEC
    if code.startswith("00"):
        return True
    # "1XXX"=North America, "3XXX"=South America, "4XXX"=Europe,
    # "5XXX"=Asia, "7XXX"=Africa, etc.
    if "X" in code.upper():
        return True
    return False


# ── Main Computation ─────────────────────────────────────────────────────────

def fetch_import_shares(
    years: Optional[list[int]] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    materials: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch US import shares by source country for all mapped materials.

    Parameters
    ----------
    years : list of int, optional
        Years to average over. Default: [2020, 2021, 2022, 2023].
    api_key : str, optional
        Census API key for higher rate limits.
    use_cache : bool
        If True, load from / save to local cache file.
    materials : list of str, optional
        Subset of materials to fetch. Default: all 25.

    Returns
    -------
    pd.DataFrame
        Columns: [material, country, share]
        where share is percentage (0-100) of total import value.
        One row per (material, country) pair.
        Compatible with all downstream pipeline consumers.
    """
    if years is None:
        years = DEFAULT_YEARS
    if materials is None:
        materials = ALL_MATERIALS

    # Try cache first
    if use_cache and CACHE_FILE.exists():
        try:
            cached = pd.read_json(CACHE_FILE)
            if set(materials).issubset(set(cached["material"].unique())):
                print("  Loaded import shares from cache.")
                result = cached[cached["material"].isin(materials)].copy()
                return result[["material", "country", "share"]]
        except Exception:
            pass  # Cache corrupt, re-fetch

    print(f"  Fetching import shares from Census Bureau API ({len(materials)} materials, "
          f"years {min(years)}-{max(years)})...")

    all_rows = []

    for mat in materials:
        codes = HTS_CODES.get(mat, [])
        if not codes:
            print(f"    WARNING: No HTS codes mapped for {mat}")
            continue

        # Query all codes for this material
        mat_data = []
        for code in codes:
            print(f"    Querying {mat}: HTS {code}...")
            df = _query_census_api(code, years, api_key)
            if not df.empty:
                mat_data.append(df)

        if not mat_data:
            print(f"    WARNING: No data returned for {mat}")
            continue

        # Combine across codes and years
        combined = pd.concat(mat_data, ignore_index=True)

        # Exclude aggregate/world/trade-bloc rows (keep individual countries only)
        combined = combined[~combined["country_code"].astype(str).apply(_is_aggregate_code)]

        # Normalize country names
        combined["country"] = combined["country"].apply(_normalize_country)

        # Aggregate: sum values by country across all codes and years,
        # then compute average annual value per country
        n_years = len(years)
        country_totals = (
            combined.groupby("country")["value"]
            .sum()
            .div(n_years)  # average annual
        )

        # Drop zero/negative
        country_totals = country_totals[country_totals > 0]

        if country_totals.empty:
            print(f"    WARNING: All zero imports for {mat}")
            continue

        # Compute percentage shares
        total_imports = country_totals.sum()
        shares = (country_totals / total_imports * 100).round(2)

        # Keep countries with ≥0.5% share (top sources),
        # aggregate the rest as "other"
        significant = shares[shares >= 0.5]
        other_share = shares[shares < 0.5].sum()

        for country, share in significant.items():
            all_rows.append({
                "material": mat,
                "country": country,
                "share": share,
            })

        if other_share > 0.5:
            all_rows.append({
                "material": mat,
                "country": "other",
                "share": round(other_share, 2),
            })

        n_countries = len(significant) + (1 if other_share > 0.5 else 0)
        print(f"    {mat}: {n_countries} source countries, "
              f"top={significant.idxmax()} ({significant.max():.1f}%)")

    result = pd.DataFrame(all_rows)

    # Save cache
    if use_cache and not result.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_json(CACHE_FILE, orient="records", indent=2)
        print(f"  Cached import shares to {CACHE_FILE}")

    return result[["material", "country", "share"]]


# ── Validation ───────────────────────────────────────────────────────────────

# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch US import shares from Census Bureau API"
    )
    parser.add_argument("--api-key", help="Census API key (optional)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch, ignore cache")
    parser.add_argument("--years", nargs="+", type=int,
                        default=DEFAULT_YEARS,
                        help="Years to average (default: 2020-2023)")
    parser.add_argument("--materials", nargs="+",
                        help="Subset of materials (default: all)")
    args = parser.parse_args()

    shares = fetch_import_shares(
        years=args.years,
        api_key=args.api_key,
        use_cache=not args.no_cache,
        materials=args.materials,
    )

    print(f"\nResult: {len(shares)} rows, {shares['material'].nunique()} materials")
    print(shares.to_string(index=False))
