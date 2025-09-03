import pandas as pd

def cleanData(df: pd.DataFrame): #ovo smo stavili da mozemo imati intellesence na varijabli df, govorimo pyu da je tip objekta DataFrame
    df.astype(str)

    for col in df.select_dtypes(include=['object']).columns: #za svaku kolonu u excelu koja je string
         df[col] = df[col].fillna(df[col].mode()[0]) #pronadji u koloni koja je najcesce koristena vrijednost, nesto ko median za brojeve
         #ova for petlja popunjava sva prazna polja gdje su stringovi, a ne brojevi

    
    for col in df.select_dtypes(include=['number']).columns: #za svaku kolonu u excelu gdje su brojevi 
         df[col] = df[col].fillna(df[col].median())  #pronadji u koloni najcescu vrijednost
        #ova for petlja popunjava sva prazna polja gdje su brojevi
    return df




