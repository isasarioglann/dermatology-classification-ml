import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier

# Dosya yükle
filename = 'KEEL_dermatology.csv'  # CSV dosyasının adı
data = pd.read_csv(filename)

# Özellikler ve etiket ayır
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Etiketleri sayısallaştır
if isinstance(Y[0], str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y = le.fit_transform(Y)

# XGBoost için etiketleri sıfırdan başlat
Y = Y - 1  # Sınıfları 1'den 0'a kaydırıyoruz

# Veri boyutları kontrolü
print(f"X boyutu: {X.shape}")
print(f"Y boyutu: {Y.shape}")
if X.shape[0] != len(Y):
    raise ValueError("Boyut uyuşmazlığı: Y, X ile aynı satır sayısına sahip olmalıdır.")

# Sıfır varyanslı özellikleri sil
zero_variance_columns = []
for col in range(X.shape[1]):
    if np.var(X[:, col]) == 0:
        zero_variance_columns.append(col)

X = np.delete(X, zero_variance_columns, axis=1)
print(f"Silinen sütun sayısı: {len(zero_variance_columns)}")

# Federated Learning parametreleri
clients = ['SVM', 'KNN', 'DT', 'NB', 'RF', 'MLR', 'XGBoost']  # Modeller
kFolds = 5  # Çapraz doğrulama kat sayısı
numClasses = len(np.unique(Y))  # Sınıf sayısı
maxRounds = 50  # Maksimum round sayısı

outputDir = 'results_federated'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# Modellerin tanımlanması
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_estimators=50),
    'MLR': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(
        
        eval_metric='mlogloss',
        n_estimators=50,  # Eğitim adımı sayısı
        
    )
}

# Çapraz doğrulama
for model_name in clients:
    model = models[model_name]
    print(f'\n🔹 Model: {model_name}')
    
    # 5 katlı çapraz doğrulama
    skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=42)
    metrics_all = []  # Her model için metrikler listesi

    # ROC Eğrisinin Ortalamasını Saklamak İçin
    mean_Xroc = []
    mean_Yroc = []

    for train_idx, test_idx in skf.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Model eğitimi
        model.fit(X_train, Y_train)
        
        # Tahmin
        Y_pred = model.predict(X_test)
        try:
            scores = model.predict_proba(X_test)
        except AttributeError:
            scores = np.zeros_like(Y_pred)

        # Confusion Matrix
        cm = metrics.confusion_matrix(Y_test, Y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Yüzde hesaplama

        # Metrikler
        TP = np.diag(cm)  # Gerçek pozitifler (köşegen)
        FP = cm.sum(axis=0) - TP  # Yanlış pozitifler
        FN = cm.sum(axis=1) - TP  # Yanlış negatifler
        TN = cm.sum() - (TP + FP + FN)  # Gerçek negatifler

        # Metrik hesaplamaları
        TPR = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)  # Gerçek Pozitif Oranı (Sensitivity)
        SPC = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)  # Özgüllük (Specificity)
        PPV = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)  # Pozitif Prediktif Değer (Precision)
        NPV = np.divide(TN, TN + FN, out=np.zeros_like(TN, dtype=float), where=(TN + FN) != 0)  # Negatif Prediktif Değer (Negative Predictive Value)
        FPR = np.divide(FP, FP + TN, out=np.zeros_like(FP, dtype=float), where=(FP + TN) != 0)  # Yanlış Pozitif Oranı (False Positive Rate)
        FNR = np.divide(FN, FN + TP, out=np.zeros_like(FN, dtype=float), where=(FN + TP) != 0)  # Yanlış Negatif Oranı (False Negative Rate)
        ACC = np.divide(TP + TN, TP + TN + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + TN + FP + FN) != 0)  # Doğru Tahmin Oranı (Accuracy)

        # Matthew's Korelasyon Katsayısı için sıfır bölmesi kontrolü ekle
        MCC = np.divide((TP * TN - FP * FN), 
                        np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 
                        out=np.zeros_like(TP, dtype=float), where=((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0)

        # F1 Skoru hesaplaması
        FM = np.divide(2 * PPV * TPR, PPV + TPR, out=np.zeros_like(PPV, dtype=float), where=(PPV + TPR) != 0)  # F1 Score (F Measure)

        metrics_all.append([TPR.mean(), SPC.mean(), PPV.mean(), NPV.mean(), FPR.mean(), FNR.mean(), ACC.mean(), MCC.mean(), FM.mean()])

        # ROC Eğrisini biriktir
        for i in range(numClasses):
            true_label = (Y_test == i)
            if len(scores.shape) > 1 and scores.shape[1] > i:  # Eğer 'predict_proba' varsa
                class_score = scores[:, i]
            else:
                class_score = np.double(Y_pred == i)  # Eğer 'predict_proba' yoksa tahminle
            if np.any(true_label):  # Pozitif örnek var mı?
                fpr, tpr, thresholds = metrics.roc_curve(true_label, class_score)
                mean_Xroc.append(fpr)
                mean_Yroc.append(tpr)

    metrics_all = np.array(metrics_all)  # Numpy array'e dönüştür
    print(f"Ortalama Metrikler: {metrics_all.mean(axis=0)}")

    # ROC Eğrisinin Ortalaması
    if mean_Xroc and mean_Yroc:  # Eğer ROC verisi varsa
        # ROC eğrisinin her bir elemanını aynı boyutta normalize et
        min_len = min(len(x) for x in mean_Xroc)
        mean_Xroc_avg = np.mean([x[:min_len] for x in mean_Xroc], axis=0)
        mean_Yroc_avg = np.mean([y[:min_len] for y in mean_Yroc], axis=0)
    
        plt.plot(mean_Xroc_avg, mean_Yroc_avg, label=f'Mean ROC - {model_name}')
        plt.title(f'Ortalama ROC - {model_name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid(True)
        plt.savefig(f'{outputDir}/{model_name}_Ortalama_ROC.png')
        plt.close()

    # Model adıyla dosya kaydetme
    metrics_df = pd.DataFrame(metrics_all, columns=['TPR', 'SPC', 'PPV', 'NPV', 'FPR', 'FNR', 'ACC', 'MCC', 'FM'])

    # Sonuçları Excel dosyasına kaydet
    metrics_df.to_excel(f'{outputDir}/{model_name}_metrics.xlsx', index=False)

print("Tüm işlemler tamamlandı.")
