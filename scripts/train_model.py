import pandas as pd
import pickle 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def train_lg_model(df: pd.DataFrame, model_path, targetColumn):
    df = df.astype(str)
    labelEncoders = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder() #pravimo labelEncoder, LabelEncoder pretvara tekst u brojeve
        df[col] = le.fit_transform(df[col]) #nauči sve različite vrijednosti u toj koloni i odmah ih pretvori u brojeve, npr crvena plava zuta crvena ce biti 0 1 2 0 
        labelEncoders[col] = le #labelEncoders pamti kako smo mapirali neku kolonu da zna i za ubuduce kako da se ponasa kada stignu novi podaci tipa ako mi dodamo sada zutu plavu da ih on automatski pohrani kao 2 1

    x = df.drop(targetColumn, axis=1) #nakon sto smo istrenirali sada sklanjamo nasu kolonu iz dataseta, axis=1 brise citavu kolonu 
    y = df[targetColumn] #istreniranu kolonu prebacujemo u y

    scaler = StandardScaler() #Njegova uloga je da sve kolone budu u sličnom rasponu.
    x_scaled = scaler.fit_transform(x) #svaka kolona u x_scaled ima prosjek 0 i standardnu devijaciju 1, sve se vrti oko toga da se nadje median, neka srednja vrijednost unutar kolone i da se na na osnovu nje odredi odstupanje drugih celija kolone

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=24) 
    #train_test_split ce uzeti 80 posto istreniranih podataka i staviti ih u x_train i y_train, 
    # a x_test i y_test je 20% (test_size = 0.2) naseg dataseta koji mozemo mi koristiti za provjeru modela

    model = LogisticRegression(max_iter=2000, random_state=24) #sve se vrti oko ovoga, pripremamo model da se trenira i da se vrti treniranje kroz podatke 2000 puta
    model.fit(x_train, y_train) #trenira onih 80% podataka
    
    with open(model_path, 'wb') as file: #otvara lg_model_path, i pohranjuje istrenirane podatke (wb - write binary)
        pickle.dump(model, file) #uzima moodel i snimi ga u file

    return model, x_test, y_test, labelEncoders


def train_rf_model(df : pd.DataFrame, model_path, targetColumn):
    df = df.astype(str)

    labelEncoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        labelEncoders[col] = le

    x = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
    
    model = RandomForestClassifier(random_state=45)
    model.fit(x_train, y_train)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    return model, x_test, y_test, labelEncoders