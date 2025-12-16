
import pandas as pd
import json, re, unicodedata
from pathlib import Path


INPUT_CSV  = "../map-labels/labels_mapped.csv"        
RULES_JSON = "rules.json"           
         
OUTPUT_CSV = "labels_mapped_normalized.csv" 
REPORT_DIR = "reports"                       



DEFAULT_RULES = [
    {"pattern": r"\bcoinbase\b|coinbase (deposit|deposits|withdrawal|withdrawals)", "entity": "Coinbase"},
    {"pattern": r"\bbinance\b|binance (deposit|deposits|withdrawal|withdrawals)", "entity": "Binance"},
    {"pattern": r"\bkraken\b", "entity": "Kraken"},
    {"pattern": r"\bgate\.?io\b|\bgate io\b|\bgate-?io\b", "entity": "GateIO"},
    {"pattern": r"\bokx\b|\bokex\b", "entity": "OKX"},
    {"pattern": r"\bbitfinex\b", "entity": "Bitfinex"},
    {"pattern": r"\bbitstamp\b", "entity": "Bitstamp"},
    {"pattern": r"\bbitpanda\b|bitpanda (deposit|deposits|withdrawal|withdrawals)", "entity": "Bitpanda"},
    {"pattern": r"\bbitmart\b", "entity": "Bitmart"},
    {"pattern": r"\bmexc\b", "entity": "MEXC"},
    {"pattern": r"\bkucoin\b", "entity": "KuCoin"},
    {"pattern": r"\bupbit\b|upbit (deposit|deposits|withdrawal|withdrawals)", "entity": "Upbit"},
    {"pattern": r"\bparibu\b", "entity": "Paribu"},
    {"pattern": r"\bgopax\b", "entity": "GOPAX"},
    {"pattern": r"\bexmo\b", "entity": "EXMO"},
    {"pattern": r"\bnaobtc\b", "entity": "NaoBTC"},
    {"pattern": r"blockchain\.com|blockchain\.com interest", "entity": "Blockchain.com"},
    {"pattern": r"\bmercado bitcoin\b", "entity": "Mercado Bitcoin"},
    {"pattern": r"\bindodax\b", "entity": "IndoDax"},
    {"pattern": r"\bokcoin\b", "entity": "Okcoin"},
    {"pattern": r"\bzebpay\b", "entity": "Zebpay"},
    {"pattern": r"\bcex\.?io\b|\bcex io\b", "entity": "CEX.IO"},
    {"pattern": r"\bwirex\b|wirex (deposit|deposits|withdrawal|withdrawals)", "entity": "Wirex"},
    {"pattern": r"\buphold\b", "entity": "Uphold"},
    {"pattern": r"\bbitvavo\b", "entity": "Bitvavo"},
    {"pattern": r"\bgoodx\b", "entity": "GoodX"},
    {"pattern": r"\bluckybird\b", "entity": "Luckybird"},
    {"pattern": r"\bsyklo\b", "entity": "Syklo"},
    {"pattern": r"\bripplefox\b", "entity": "RippleFox"},
    {"pattern": r"\bcentre\b", "entity": "Centre"},

    {"pattern": r"\blobstr\b|lobstr assets custodian|lobstr merge tool", "entity": "Lobstr"},
    {"pattern": r"\bvibrant\b", "entity": "Vibrant"},
    {"pattern": r"\bpapaya ?bot\b|\bpapaya\b", "entity": "PapayaBot"},
    {"pattern": r"\bsideshift\b", "entity": "SideShift"},
    {"pattern": r"\bstormgain\b", "entity": "StormGain"},
    {"pattern": r"\banchorusd\b", "entity": "AnchorUSD"},
    {"pattern": r"\baqua\b|aqua (mm rewards|issuer|airdrop)", "entity": "AQUA"},
    {"pattern": r"\btmm ?bot\b", "entity": "TMM bot"},
    {"pattern": r"\bexlm\b", "entity": "EXLM"},
    {"pattern": r"\bsdf\b|lightyear option agreement|ecosystem support|user acquisition|developer support|direct development|early employee grants", "entity": "SDF"},

    {"pattern": r"\bburn account\b", "entity": "Burn Account"},
    {"pattern": r"\bspam issuer\b", "entity": "Spam Issuer"},
    {"pattern": r"\bimposter\b", "entity": "Imposter"},
    {"pattern": r"\bfirefly\b", "entity": "Firefly"},

    {"pattern": r"scam|minter|counterfeit(er)?|counterfeiter|phish|launder|laundering|air[- ]?drop|fraud|spam|"
                 r"scam[- ]?minter|serial scam|scam[- ]?counterfeit|scam[- ]?counterfeiter",
     "entity": "SCAM"},
]


def load_rules():
    rules_path = Path(RULES_JSON)
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return DEFAULT_RULES

def base_norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u0000-\u001f\u007f]", " ", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("desposits", "deposits")
    s = s.replace("countereiter", "counterfeiter")
    s = s.replace("counterfieter", "counterfeiter")
    s = s.replace("okex", "okx")
    s = s.replace("air drop", "air-drop")
    s = re.sub(r"[,\.;:!]+$", "", s)
    return s

def canon_name(n_raw: str, n_norm: str, compiled) -> str:
    for pat, ent in compiled:
        if pat.search(n_norm):
            return ent
    return n_raw.strip()


def main():
    df = pd.read_csv(INPUT_CSV).dropna(subset=["account_id","name"]).drop_duplicates()

    rules = load_rules()
    compiled = [(re.compile(r["pattern"], re.IGNORECASE), r["entity"]) for r in rules]

    df["__norm"] = df["name"].map(base_norm)
    df["name"] = [canon_name(r["name"], r["__norm"], compiled) for _, r in df.iterrows()]
    out = df[["account_id","name"]].drop_duplicates()
    out.to_csv(OUTPUT_CSV, index=False)

    # Reports
    report_dir = Path(REPORT_DIR); report_dir.mkdir(exist_ok=True)
    entity_counts = out["name"].value_counts().reset_index()
    entity_counts.columns = ["entity","n_accounts"]
    entity_counts.to_csv(report_dir / "entity_account_counts.csv", index=False)

    print(f"Saved normalized entities -> {OUTPUT_CSV}")
    print(f"Rows in: {len(df)} | Rows out: {len(out)} | Unique entities: {out['name'].nunique()}")

if __name__ == "__main__":
    main()
