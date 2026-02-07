import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

print(f"Total emails: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nLabels: {df['label'].unique()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nFirst 3 emails:")
for i in range(3):
    print(f"\n{i+1}. Label: {df.iloc[i]['label']}")
    print(f"   Text: {df.iloc[i]['text'][:80]}...")
