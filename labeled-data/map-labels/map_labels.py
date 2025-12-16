import pandas as pd
import json
from datetime import datetime



LABELS = '../full_stellar_directory.json'
METADATA = '~/stellar-clustering/publication/data/transaction_edges_metadata.csv'


with open(LABELS, 'r') as f:
    labels_data = json.load(f)


address_to_label = {}
for item in labels_data:
    address = item['address']
    name = item.get('name', '')
    if name:
        address_to_label[address] = name

print(f"Loaded {len(address_to_label)} labeled addresses")





df = pd.read_csv(METADATA)
print(f"Loaded {len(df)} transactions")

# get all accounts sender and receiver
all_accounts = pd.concat([
    df[['sender_id', 'sender_address']].rename(columns={'sender_id': 'account_id', 'sender_address': 'address'}),
    df[['receiver_id', 'receiver_address']].rename(columns={'receiver_id': 'account_id', 'receiver_address': 'address'})
]).drop_duplicates()

print(f"Found {len(all_accounts)} unique accounts")


all_accounts['label'] = all_accounts['address'].map(address_to_label)

labeled_accounts = all_accounts[all_accounts['label'].notna()][['account_id', 'label']]

print(f"Mapped {len(labeled_accounts)} labeled accounts")




labeled_accounts.to_csv('labels_mapped.csv', index=False)

print(f"saved")