from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(model, x_test, y_test, encoders):
    y_pred = model.predict(x_test) #na osnovu modela (istreniranih podataka (80%)) prediktuj y test na osnovu x_test, koristi model i koristi x_test da prediktuje y_predicted
    #ako je x_test visina koja iznosi 180, i y_pred je tezina koja treba biti predicotvana,
    #  .predict ce otici u model i naci najblizu vrijednost x_testu (visini 180) i na osnovu modela, u kojem je tipa pohranjeno da neko sa 180 visinom ima 78 kila, odrediti i tezinu za y_pred
    
    targetEncoder = encoders['dcNyha'] #iz encodera uzimamo zeljenu kolonu, dcNyha
    y_pred = targetEncoder.inverse_transform(y_pred) #svi zapisi su u modelu u brojevima, moramo ih uz pomoc encodera, vratiti u origirale npr stringove, iz 0 u crvenu i sl
    y_test = targetEncoder.inverse_transform(y_test)

    performance = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_test, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=1)
    }

    return performance