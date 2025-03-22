import pandas as pd

seeds_data = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/seeds.csv")

#make it look like an ADNI dataset
seeds_data.rename(columns={"Type": "LAST_DX"}, inplace=True)
seeds_data["RID"] = [x + 2 for x in range(len(seeds_data))]

print(seeds_data)