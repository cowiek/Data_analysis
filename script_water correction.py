import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_mean_intensity(filepath):
    # Wczytanie pliku CSV, pomijając pierwsze 3 wiersze
    df = pd.read_csv(filepath, skiprows=3)
    
    # Wybór kolumn intensywności: 4, 8, 12, 16, ...
    intensity_cols = [i for i in range(1, df.shape[1]) if (i - 3) % 4 == 0]
    
    # Pobranie danych intensywności
    intensities = df.iloc[:, intensity_cols].apply(pd.to_numeric, errors='coerce')

    # Średnia intensywność wszystkich kolumn
    mean_intensity = intensities.mean(axis=1)
    
    # Wavenumber – bierzemy tylko pierwszą kolumnę (np. kolumna 3)
    wavenumbers = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    
    # Obliczenie absorbancji wody (np. stosunek pierwszych 10 i kolejnych 10 pomiarów)
    intensity_1_10 = intensities.iloc[:, :10]
    intensity_11_20 = intensities.iloc[:, 10:20]
    epsilon = 1e-10
    ratio = (intensity_1_10.mean(axis=1) + epsilon) / (intensity_11_20.mean(axis=1) + epsilon)
    water_absorbance = -np.log(ratio)
    
    return wavenumbers, mean_intensity, water_absorbance

def calculate_absorbance(p0_file, p90_file):
    wavenumbers_p0, intensity_p0, water_absorbance_p0 = get_mean_intensity(p0_file)
    wavenumbers_p90, intensity_p90, water_absorbance_p90 = get_mean_intensity(p90_file)


    if not np.allclose(wavenumbers_p0, wavenumbers_p90, equal_nan=True):
        raise ValueError("Wavenumbers in p0 and p90 files do not match!")

    epsilon = 1e-10
    ratio = (intensity_p90 + epsilon) / (intensity_p0 + epsilon)
    absorbance = -np.log(ratio)

    avg_water_absorbance = (water_absorbance_p0 + water_absorbance_p90) / 2

    return pd.DataFrame({
        'Wavenumber': wavenumbers_p0,
        'Absorbance': absorbance,
        'Water_Absorbance': avg_water_absorbance
    })

# 🔄 Podmień daty plików na aktualne
file_pairs = {
    "BR_TN_IR": {
        "p0": r'C:\Users\jskub\OneDrive\Pulpit\Studies\Twente\master_thesis\Experimental\IR\IR analysis\20251021_fifth\imported_from_origin\BR_20250617_TN_IR_p0.csv',
        "p90": r'C:\Users\jskub\OneDrive\Pulpit\Studies\Twente\master_thesis\Experimental\IR\IR analysis\20251021_fifth\imported_from_origin\BR_20250617_TN_IR_p90.csv'
    },
    "CAR_TN_IR": {
        "p0": r'C:\Users\jskub\OneDrive\Pulpit\Studies\Twente\master_thesis\Experimental\IR\IR analysis\20251021_fifth\imported_from_origin\CAR_20250617_TN_IR_p0.csv',
        "p90": r'C:\Users\jskub\OneDrive\Pulpit\Studies\Twente\master_thesis\Experimental\IR\IR analysis\20251021_fifth\imported_from_origin\CAR_20250617_TN_IR_p90.csv'
    }
}

x_car = 0  # Współczynnik korekcji na wodę
x_br = 0

results = {}
for sample_name, files in file_pairs.items():
    df_abs = calculate_absorbance(files["p0"], files["p90"])
    results[sample_name] = df_abs

df_car = results["CAR_TN_IR"]
df_br = results["BR_TN_IR"]

if not np.allclose(df_car['Wavenumber'], df_br['Wavenumber'], equal_nan=True):
    raise ValueError("Wavenumbers do not match between CAR_TN_IR and BR_TN_IR!")

# ✅ Korekcja wody dla pojedynczych widm
df_car['Corrected_Absorbance'] = df_car['Absorbance'] - x_car * df_car['Water_Absorbance']
df_br['Corrected_Absorbance'] = df_br['Absorbance'] - x_br * df_br['Water_Absorbance']

# ✅ Następnie różnica widm już skorygowanych
differential_absorbance = df_car['Corrected_Absorbance'] - df_br['Corrected_Absorbance']

# (jeśli chcesz, możesz też zachować uśrednione Water_Absorbance tylko dla informacji)
avg_water_absorbance = (df_car['Water_Absorbance'] + df_br['Water_Absorbance']) / 2

corrected_df = pd.DataFrame({
    'Wavenumber': df_car['Wavenumber'],
    'Differential_Absorbance': differential_absorbance,
    'Water_Absorbance': avg_water_absorbance,
})

#Differential spectrum
plt.figure(figsize=(10,6))
plt.plot(corrected_df['Wavenumber'], corrected_df['Differential_Absorbance'],
         label='Absorbance (CAR_TN_IR - BR_TN_IR) - water correction factor')
plt.xlabel('Wavenumber [cm⁻¹]')
plt.ylabel('Corrected Absorbance [-]')
plt.title('Water Corrected Absorbance CAR_TN_IR - BR_TN_IR')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
start_tick = 400
end_tick = int(corrected_df['Wavenumber'].max()) + 1
plt.xticks(np.arange(start_tick, end_tick, 200))
plt.show(block=False)


#Individual spectra
plt.figure(figsize=(10, 6))

offset_car = 0.0  # you can adjust this
offset_br = 0

plt.plot(df_car['Wavenumber'], df_car['Corrected_Absorbance'] + offset_car, label='After reaction', color='tab:blue')
plt.plot(df_br['Wavenumber'], df_br['Corrected_Absorbance'] + offset_br, label='Before reaction ', color='tab:orange')

plt.xlabel('Wavenumber [cm⁻¹]')
plt.ylabel('Absorbance [-]')
plt.title('')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
start_tick = 400
end_tick = int(corrected_df['Wavenumber'].max()) + 1
plt.xticks(np.arange(start_tick, end_tick, 200))
plt.show()

# #Save corrected (differential) absorbance to .txt
# corrected_df[['Wavenumber', 'Differential_Absorbance']].to_csv(
#     "C:/Users/jskub/OneDrive/Pulpit/Studies/Twente/master_thesis/Experimental/IR/IR analysis/20251021_fifth/results/differential_spectrum.txt",
#     sep='\t',
#     index=False,
#     header=['Wavenumber [cm⁻¹]', 'Corrected Absorbance [-]']
# )

# # # Save individual absorbance spectra to .txt
# # df_car[['Wavenumber', 'Corrected_Absorbance']].to_csv(
# #     "C:/Users/jskub/OneDrive/Pulpit/Studies/Twente/master_thesis/Experimental/IR/IR analysis/20251021_fifth/results/CAR_TN_IR_absorbance.txt",
# #     sep='\t',
# #     index=False,
# #     header=['Wavenumber [cm⁻¹]', 'Absorbance [-]']
# # )

# # df_br[['Wavenumber', 'Absorbance']].to_csv(
# #     "C:/Users/jskub/OneDrive/Pulpit/Studies/Twente/master_thesis/Experimental/IR/IR analysis/20251021_fifth/results/BR_TN_IR_absorbance.txt",
# #     sep='\t',
# #     index=False,
# #     header=['Wavenumber [cm⁻¹]', 'Absorbance [-]']
# # )

# print("done")
