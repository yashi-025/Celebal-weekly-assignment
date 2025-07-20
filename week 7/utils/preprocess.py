import pandas as pd

def prepare_input(carat, depth, table):
    return pd.DataFrame([[carat, depth, table]], columns=['carat', 'depth', 'table'])
