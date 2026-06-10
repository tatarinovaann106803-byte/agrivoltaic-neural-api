import pandas as pd

print("Очистка данных от NaN и преобразование процентов...")

for sector in ['crop_farming', 'aquaculture', 'forestry']:
    file_path = f'dataset/sector_data/{sector}.csv'
    try:
        df = pd.read_csv(file_path)
        before = len(df)
        
        # Удаляем строки с NaN
        df = df.dropna()
        
        # Обрабатываем колонку с процентами (если есть)
        if 'productivity_change' in df.columns:
            # Преобразуем в строку, удаляем %, заменяем запятые на точки, конвертируем в float
            df['productivity_change'] = df['productivity_change'].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
            print(f'{sector}: обработаны проценты в колонке productivity_change')
        
        # Обрабатываем все числовые колонки на всякий случай
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
                print(f'{sector}: колонка {col} преобразована в числа')
            except:
                pass  # оставляем как есть, если не получается
        
        after = len(df)
        df.to_csv(file_path, index=False)
        print(f'{sector}: {before} -> {after} строк (удалено {before - after})')
        
    except Exception as e:
        print(f'Ошибка при очистке {sector}: {e}')

print("Очистка завершена!")
