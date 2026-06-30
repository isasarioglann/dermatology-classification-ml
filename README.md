# dermatology-classification-ml

# 🩺 Dermatolojik Hastalıkların Sınıflandırılması ve ML Model Yarıştırma Projesi

Bu proje, dermatoloji veri kümesi (`KEEL_dermatology.csv`) üzerindeki klinik özellikleri analiz ederek cilt hastalıklarının teşhis edilmesini amaçlayan gelişmiş bir **Makine Öğrenmesi (Machine Learning)** projesidir. Proje kapsamında, Federated Learning (Federasyonel Öğrenme) simülasyon altyapısı kullanılarak 7 farklı popüler yapay zeka modeli birbiriyle yarıştırılmaktadır.

## 📊 Yarıştırılan Yapay Zeka Modelleri
Proje, verileri bağımsız istemciler (clients) gibi ele alarak aşağıdaki 7 algoritmayı eğitir ve test eder:
1. **SVM** (Support Vector Machine)
2. **KNN** (K-Nearest Neighbors)
3. **DT** (Decision Tree)
4. **NB** (Gaussian Naive Bayes)
5. **RF** (Random Forest)
6. **MLR** (Multinomial Logistic Regression)
7. **XGBoost** (Extreme Gradient Boosting)

## 🛠 Proje Mimarisi ve Metotlar
- **Veri Önişleme:** Varyansı sıfır olan (ayırt edici özelliği bulunmayan) sütunlar otomatik olarak tespit edilerek veri setinden temizlenir. Etiketler `LabelEncoder` ile normalize edilir.
- **5-Fold Stratified Cross-Validation:** Veri kümesi, sınıf dağılımları korunarak 5 farklı kata ayrılır. Her model 5 kez eğitilerek aşırı öğrenmenin (overfitting) önüne geçilir.
- **Klinik Değerlendirme Metrikleri:** Modeller sadece doğruluk oranıyla değil, tıbbi teşhiste kritik öneme sahip 9 farklı metrikle (TPR/Sensitivity, SPC/Specificity, PPV, NPV, FPR, FNR, ACC, MCC, FM) değerlendirilir.
- **ROC Analizi ve Raporlama:** Her modelin eğitimi bittiğinde, performans grafiklerini içeren **Mean ROC Curve** görseli (`.png`) ve detaylı metrik tablosu (`.xlsx`) otomatik olarak dışa aktarılır.

## 📦 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin:
   ```bash
   git clone [https://github.com/kullanici-adiniz/dermatology-classification-ml.git](https://github.com/kullanici-adiniz/dermatology-classification-ml.git)
   cd dermatology-classification-ml

   Gerekli kütüphaneleri yükleyin:

Bash
pip install -r requirements.txt
Modelleri yarıştırmak ve grafikleri üretmek için kodu çalıştırın:

Bash
python main.py

📈 Çıktılar
Kod çalıştırıldıktan sonra kök dizinde results_federated/ adında bir klasör oluşur. Bu klasörün içerisinde her modele ait ROC eğrisi grafiği ve fold bazlı başarı metriklerini içeren Excel dosyaları yer almaktadır.
