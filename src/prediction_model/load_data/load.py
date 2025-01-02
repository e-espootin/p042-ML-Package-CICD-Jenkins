import clickhouse_driver
import pandas as pd
from prediction_model.config.clickhouse_config import CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DB
from sklearn.model_selection import train_test_split
import os
from typing import Tuple


class ClickhouseDataLoader:
    def __init__(self):
        self.connection = clickhouse_driver.Client(
            host=CLICKHOUSE_HOST,
            # port=port,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DB
        )

    def load_data(self, query) -> pd.DataFrame:
        data = self.connection.execute(query, with_column_types=True)
        columns = [col[0] for col in data[1]]
        df = pd.DataFrame(data[0], columns=columns)
        # add column is_fraud to the dataframe
        # df['fraud'] = 0
        df['fraud'] = df.apply(lambda x: 1 if x['amount']
                               > 9000000
                               or x['amount'] < 3
                               or x['location'] > 900 else 0, axis=1)
        return df

    def split_data(self, df, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    def save_dataframes(self, train_df, test_df, dataset_dir):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        train_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False)
