from data_source.data_reader import DataReader, register_csv_dialect

print('Loading data...')
register_csv_dialect()
dr = DataReader()
dr.read_data()
print('Data prepared')


def get_datasource():
    return dr.data