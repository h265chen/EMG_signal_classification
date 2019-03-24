import pandas as pd
class lift_file:
    file_path = None

    def __init__(self,path):
        self.file_path = path

    def printFileName(self):
        print(self.file_path)

    def get_df(self):
        df = pd.read_csv(self.file_path, sep='\t', header=None)
        df.columns = ['BB', 'TB', 'BR', 'AD', 'LES', 'TES', 'hand switch', 'box switch', 'MS', 'MS', 'MS', 'MS', 'MS',
                      'MS']
        return df
