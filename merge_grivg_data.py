import pandas as pd

# Load the three datasets from the datasets folder
games_df = pd.read_csv("datasets/games.grivg.csv")
characters_df = pd.read_csv("datasets/characters.grivg.csv")
sexualization_df = pd.read_csv("datasets/sexualization.grivg.csv")

# Merge games and characters
# Games.Game_Id = Characters.Game
merged_df = characters_df.merge(
    games_df,
    left_on="Game",
    right_on="Game_Id",
    how="left"
)

# Merge with sexualization data
# Characters.Id = Sexualization.Id
merged_df = merged_df.merge(
    sexualization_df,
    on="Id",
    how="left",
    suffixes=("", "_sexualization")
)

# Save the merged file into datasets folder
merged_df.to_csv("datasets/merged_grivg_data.csv", index=False)

# Check result
print("Games shape:", games_df.shape)
print("Characters shape:", characters_df.shape)
print("Sexualization shape:", sexualization_df.shape)
print("Merged shape:", merged_df.shape)

print("Merged file saved to datasets/merged_grivg_data.csv")
print(merged_df.head())