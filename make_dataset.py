import glob
import pandas as pd

# mapeia o caminho dos dados
all_files = glob.glob("/home/CIN/vlfo/Downloads/MachineLearningCSV/*.csv")

# le os csv
df_list = [pd.read_csv(file) for file in all_files]

# junta os csv
combined_df = pd.concat(df_list, ignore_index=True)

# transforma para parquet para economizar espaco e memoria ram nas operacoes
combined_df.to_parquet("pacotes.parquet", engine="pyarrow", compression="snappy")
