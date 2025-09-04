from load_data import loadData
from clean_data import cleanData
from train_model import train_lg_model, train_rf_model, train_gb_model
from calculate_metrics import calculate_metrics
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    data_path = 'data/kardiologija_hospitalizacija.xlsx'
    lg_model_path = 'scripts/lg_model.pkl' #logistic regression
    gb_model_path = 'scripts/gb_model.pkl' #gradient boosting
    rf_model_path = 'scripts/rf_model.pkl' #random forest

    df = loadData(data_path) #otvara skriptu load_data.py i otvara excel sheet


    df = cleanData(df) #cisti kolone u tabeli

    #print(df.info())
    # print("----------------------------------------------------------------------------")
    #print(df.describe())
    # print("----------------------------------------------------------------------------")
    #print(df.isnull().sum())
    # print("----------------------------------------------------------------------------")
    # print(df.nunique())
    # print("----------------------------------------------------------------------------")
    # print(df.dtypes)
    # print("----------------------------------------------------------------------------")



    targetColumn = 'dcNyha' #tabela u excelu koju cemo obradit

    lg_trained_model, x_test, y_test, encoders = train_lg_model(df, lg_model_path, targetColumn) #saljemo joj excel podatke, mjesto gdje treba da pohrani trenirane podatke i kolonu s kojom treba da radi
    #i pohranjujemo trenirani model, x_test, y_test, i labelEncoders tj encoders => mapirane podatke
    lg_metrics = calculate_metrics(lg_trained_model, x_test, y_test, encoders)

    # #ISTO SVE RADIMO SADA SAMO ZA DRUGI MODEL
    # rf_trained_model, x_test, y_test, encoders = train_rf_model(df, rf_model_path, targetColumn)
    # rf_metrics = calculate_metrics(rf_trained_model, x_test, y_test, encoders)

    # #GRADIENT BOOSTING METODA
    # gb_trained_model, x_test, y_test, encoders = train_gb_model(df, gb_model_path, targetColumn)
    # gb_metrics = calculate_metrics(gb_trained_model, x_test, y_test, encoders)

    print("LR Metrics rezultati:")
    for k, v in lg_metrics.items(): #u lg_metrics imamo "k" koji moze biti accuracy precision recall ili f1 i "v" tj vrijednost, za svaku metriku i njenu vrijednost isprintaj...
        print(f"{k}:{v:.4f}") # formatira string, ^{k} odstampa ime metrike, odštampa vrijednost sa 4 decimale 
    print("----------------------------------------------------------------------------")



    # print("RF Metrics rezultati:")
    # for k, v in rf_metrics.items():
    #     print(f"{k}:{v:.4f}")

    # print("GB Metrics rezultati")
    # for k, v in gb_metrics.items():
    #     print(f"{k}:{v:.4f}")

if __name__ == "__main__":
    main()


