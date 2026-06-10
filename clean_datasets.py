import pandas as pd

print("Очистка данных от NaN...")

for sector in ['crop_farming', 'aquaculture', 'forestry']:
    file_path = f'dataset/sector_data/{sector}.csv'
    try:
        df = pd.read_csv(file_path)
        before = len(df)
        df = df.dropna()
        df.to_csv(file_path, index=False)
        after = len(df)
        print(f'{sector}: {before} -> {after} строк (удалено {before - after})')
    except Exception as e:
        print(f'Ошибка при очистке {sector}: {e}')

print("Очистка завершена!")