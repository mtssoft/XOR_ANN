# train.py
from xor_nn import XOR_NN
import numpy as np

# Eğitilen modeli test etmek için kullanılacak XOR problemine uygun test verisini rasgele üreten fonksiyon
def generateRandomTestData(sampleCount):
    return np.random.randint(0, 2, (sampleCount, 2))

# XOR için eğitim veri seri hazırlanır.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Eğitim veri setine ait beklenen çıkış değerleri
y = np.array([[0], [1], [1], [0]])

# XOR_NN sınıfından bir instance yaratılarak sinir ağı modeli oluşturulur.
xor_nn = XOR_NN()

# sinir ağı modelinin train fonksiyonu eğitim veri seti ile kullanılarak model eğitilir
xor_nn.train(X, y)

# test için random test verisi üretilir
test_data = generateRandomTestData(5)

print(f"\n Test Verileri ve Sonuçlar :")

# Random olarak üretilen test verisi kullanılarak tahminler üretilir. Her test verisi ve veriye ait tahmin değeri yazdırılır.
for test in test_data:
    output = xor_nn.predict(test.reshape(1, -1))  # Test verisini yeniden şekillendir
    print(f"Giriş: {test} -> Çıkış: {output[0][0]:.4f}")
