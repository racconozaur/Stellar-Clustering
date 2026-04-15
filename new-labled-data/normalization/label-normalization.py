

import csv
import logging
import re
import sys
from collections import Counter

INPUT_FILE = "../stellar_directory.csv"
OUTPUT_FILE = "stellar_directory_normalized.csv"
LOG_FILE = "normalize_labels.log"

# ---------- logging ----------
logger = logging.getLogger("normalize")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)


# ---------- rules ----------

# Any name matching any of these -> malicious, regardless of tags.
MALICIOUS_NAME_PATTERNS = [
    r"\bscam\b",
    r"counterfeit",
    r"phishing",
    r"\bfraud\b",
    r"\bfake\b",
    r"\bimposter\b",
    r"stolen\s+funds",
    r"\bspam\b",
    r"serial\s+.*minter",
    r"\bcaution\b",
    r"\bdead\b",
    r"\babandoned\b",
    r"deprecated",
    r"out of service",
    r"unsafe\s+copycat",
    r"domain for sale",
]
MALICIOUS_NAME_RE = re.compile("|".join(MALICIOUS_NAME_PATTERNS), re.IGNORECASE)

# Suffixes to strip from names when finding the parent entity.
# Applied iteratively until stable. Order matters (longest first).
SUFFIX_PATTERNS = [
    r"\s*\(Hot\s*\d*(\s*-\s*old)?\)\s*$",
    r"\s*\(Old\)\s*$",
    r"\s*\(legacy\)\s*$",
    r"\s+-\s+Discontinued$",
    r"\s+-\s+Discontiniued$",       # typo in data
    r"\s+-\s+Abandoned$",
    r"\s+-\s+Out of Service$",
    r"\s+-\s+In-App Distribution.*$",
    r"\s+ColdStorage$",
    r"\s+Coldwallet$",
    r"\s+ColdWallet$",
    r"\s+Cold\s+Storage$",
    r"\s+[Dd]eposits?$",
    r"\s+[Dd]esposits?$",            # typo in data
    r"\s+[Ww]ithdrawals?$",
    r"\s+Hot\s*\d*$",
    r"\s+Old$",
    r"\s+Router$",
    r"\s+Pool$",
    r"\s+Pool\s+Factory$",
    r"\s+Backstop$",
    r"\s+Emitter$",
    r"\s+Interest$",
    r"\s+Reserve$",
    r"\s+Signer$",
    r"\s+Swap$",
    r"\s+Issuer$",
    r"\s+Distribution$",
    r"\s+Distributor$",
    r"\s+Vault\s+Signer$",
    r"\s+Merge\s+Tool$",
    r"\s+Wallet\s+Root$",
    r"\s+Assets\s+Custodian$",
    r"\s+FeeCollector$",
]

# Family prefixes: any name starting with "AQUA ..." rolls up to "AQUA".
FAMILY_PREFIXES = [
    "AQUA",
    "SDF",
    "LMX",
    "Blend",
    "Lobstr",
    "UltraCapital",
    "BRAVE",
    "CARBON",
    "Papaya",
    "Aquarius",
    "SoroSwap",
    "Sushi",
    "Phoenix",
    "Reflector",
    "FxDAO",
    "Stellarcarbon",
    "StellarExpert",
    "Tokenized Blend",
    "Normal Finance",
    "Wirecash",
    "NGNC",
    "AnchorMXN",
    "AnchorUSD",
    "Etherfuse",
    "YieldBlox",
    "Kale",
    "SureRemit",
    "Soroban",
    "Apay",
    "AQUA",
    "xBull",
    "XBull",
    "LabIO",
    "Wirex",
    "WireX",
    "CryptoCom",
    "Zeam",
    "TDC",
    "MCP",
    "DLD",
    "R2B",
    "LibreTest",
    "LumosDAO",
    "Network Upgrade",
    "Lightyear",
    "SDF/",
]

# Explicit aliases: canonical casing / spelling for entity names.
# Applied AFTER suffix stripping and family folding, as a final cleanup.
ALIAS_MAP = {
    "Wirex": "Wirex",
    "WireX": "Wirex",
    "Allbridge": "Allbridge",
    "AllBridge": "Allbridge",
    "Wisdom Tree": "WisdomTree",
    "WisdomTree": "WisdomTree",
    "Zeam.money": "Zeam.Money",
    "Zeam.Money": "Zeam.Money",
    "Gate.io": "Gate.io",
    "GateIO": "Gate.io",
    "Kucoin": "KuCoin",
    "KuCoin": "KuCoin",
    "Circle / Centre": "Circle",
    "Centre": "Circle",
    "Circle": "Circle",
    "SoroSwap": "SoroSwap",
    "Sorobanomains": "SorobanDomains",
    "Soroban Domains": "SorobanDomains",
    "SorobanDomains": "SorobanDomains",
    "aps.money": "APS.Money",
    "APS.Money - Advanced Payment Solutions Ltd.": "APS.Money",
    "StellarQuest": "Stellar Quest",
    "Stellar Quest": "Stellar Quest",
    "Lux Payband": "LUX Payband",
    "LUX Payband": "LUX Payband",
    "Papaya": "Papaya",
    "PapayaBot": "Papaya",
    "PapayaSwap": "Papaya",
    "Blockchain.com": "Blockchain.com",
}

# Tag -> category priority. First match wins.
CATEGORY_PRIORITY = [
    "malicious",
    "exchange",
    "custodian",
    "anchor",
    "wallet",
    "issuer",
    "defi",
    "sdf",
    "application",
    "infra",
    "airdrop",
    "personal",
    "memo-required",
    "unsafe",
    "obsolete-inflation-pool",
]


def pick_category(tag_list):
    tagset = {t.strip() for t in tag_list if t.strip()}
    for cat in CATEGORY_PRIORITY:
        if cat in tagset:
            return cat
    return "other"


def strip_suffixes(name):
    prev = None
    cur = name
    # iterate until fixed point (handles "X Deposits Hot 1" etc.)
    while cur != prev:
        prev = cur
        for pat in SUFFIX_PATTERNS:
            cur = re.sub(pat, "", cur).strip()
    return cur


def fold_family(name):
    for fam in FAMILY_PREFIXES:
        # Match "AQUA" exactly or "AQUA <anything>"
        if name == fam or name.startswith(fam + " ") or name.startswith(fam + "/"):
            return fam
    return name


def normalize_entity(raw_name):
    if not raw_name:
        return "UNKNOWN"
    name = raw_name.strip()
    name = strip_suffixes(name)
    name = fold_family(name)
    name = ALIAS_MAP.get(name, name)
    return name


def is_malicious(name, tags):
    if "malicious" in tags:
        return True
    if MALICIOUS_NAME_RE.search(name or ""):
        return True
    return False


def main():
    logger.info("=" * 60)
    logger.info(f"Reading {INPUT_FILE}")

    rows_in = []
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)
    logger.info(f"  {len(rows_in)} input rows")

    rows_out = []
    entity_counter = Counter()
    category_counter = Counter()
    malicious_n = 0

    for row in rows_in:
        address = row["address"]
        name_raw = row.get("name", "") or ""
        tags_raw = row.get("tags", "") or ""
        domain = row.get("domain", "") or ""
        tag_list = [t.strip() for t in tags_raw.split(",") if t.strip()]

        if is_malicious(name_raw, tag_list):
            entity = "MALICIOUS"
            name_normalized = "SCAM"
            category = "malicious"
            malicious_n += 1
        else:
            entity = normalize_entity(name_raw)
            name_normalized = entity
            category = pick_category(tag_list)

        entity_counter[entity] += 1
        category_counter[category] += 1

        rows_out.append({
            "address": address,
            "name_normalized": name_normalized,
            "entity": entity,
            "category": category,
            "domain": domain,
            "tags_raw": tags_raw,
            "name_raw": name_raw,
        })

    logger.info(f"Writing {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "address",
                "name_normalized",
                "entity",
                "category",
                "domain",
                "tags_raw",
                "name_raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    # -------- summary --------
    logger.info("-" * 60)
    logger.info(f"Total rows out:  {len(rows_out)}")
    logger.info(f"Malicious rows:  {malicious_n}")
    logger.info(f"Unique entities: {len(entity_counter)}")
    logger.info(f"  (excluding MALICIOUS): {len(entity_counter) - 1}")
    logger.info("")
    logger.info("Category distribution:")
    for cat, n in category_counter.most_common():
        logger.info(f"  {cat:25s} {n}")
    logger.info("")
    logger.info("Top 40 entities (after normalization):")
    for ent, n in entity_counter.most_common(40):
        logger.info(f"  {ent:40s} {n}")

    # Show singletons — entities with exactly 1 address — useful to eyeball
    singletons = [e for e, n in entity_counter.items() if n == 1]
    logger.info("")
    logger.info(f"Singletons: {len(singletons)} entities with 1 address each")


if __name__ == "__main__":
    main()