from xor_nn import XOR_NN
import numpy as np
import matplotlib.pyplot as plt

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

# Eğitim sonrasında kayıp fonksiyonunu çizdirilir
plt.plot(xor_nn.epoch_losses)
plt.title('Dönemler Süresince Kayıp Grafiği')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Kayıp(Loss)')
plt.show()

# Eğitim sonrasında girdi ve gizli katman için yani nöron 1 için ağırlıkların nasıl değiştiğini
# gözlemleyebileceğimiz ağırlık seti grafiği çizdirilir
plt.plot([w[0, 0] for w in xor_nn.epoch_weights_input_hidden])
plt.title('Dönemler(Epoch) Süresince Girdi-Gizli Katman Ağırlıkları (Nöron 1)')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Ağırlık Değeri')
plt.show()

# Eğitim sonrasında girdi ve gizli katman için yani nöron 1 için ağırlıkların nasıl değiştiğini
# gözlemleyebileceğimiz ağırlık seti grafiği çizdirilir
plt.plot([w[0, 0] for w in xor_nn.epoch_weights_hidden_output])
plt.title('Dönemler(Epoch) Süresince Gizli-Çıktı Katmanı Ağırlıkları (Nöron 2)')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Ağırlık Değeri')
plt.show()

# Eğitim sonrasında gizli katman için yani nöron 1 ve 2 için için bias değerlerinin nasıl değiştiğini
# gözlemleyebileceğimiz graifk çizdirilir
plt.plot([b[0, 0] for b in xor_nn.epoch_bias_hidden], label='Bias Gizli Katman (Nöron 1)')
plt.plot([b[0, 1] for b in xor_nn.epoch_bias_hidden], label='Bias Gizli Katman (Nöron 2)')
plt.title('Dönemler (Epochs) Boyunca Gizli Katman Bias Değerleri')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Bias Değeri')
plt.legend()
plt.show()

# Eğitim sonrasında çıkış katmanı bias değerlerinin değişimi
plt.plot([b[0, 0] for b in xor_nn.epoch_bias_output], label='Bias Çıktı Katmanı')
plt.title('BDönemler Boyunca Çıktı Katmanı Bias Değerleri')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Bias Değeri')
plt.legend()
plt.show()

# Eğitim sonrası gizli katman delta değerleri grafiği çizdirilir
delta_hidden_layer_0 = [np.mean(d[:, 0]) for d in xor_nn.epoch_delta_hidden_layer]
delta_hidden_layer_1 = [np.mean(d[:, 1]) for d in xor_nn.epoch_delta_hidden_layer]

plt.plot(delta_hidden_layer_0, label="Gizli Katman Nöron 1 Delta")
plt.plot(delta_hidden_layer_1, label="Gizli Katman Nöron 2 Delta")
plt.title('Dönemler Süresince Gizli Katman Delta Değerleri')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Delta Değerleri')
plt.legend()
plt.show()

# Eğitim sonrasında çıkış katmanı delta değerleri grafiği çizdirilir
delta_output_layer = [np.mean(d) for d in xor_nn.epoch_delta_output_layer]

plt.plot(delta_output_layer, label="Çıktı Katmanı Delta", color='red')
plt.title('Dönemler Boyunca Çıktı Katmanı Delta Değerleri')
plt.xlabel('Dönemler(Epochs)')
plt.ylabel('Delta Değeri')
plt.legend()
plt.show()



# test için random test verisi üretilir
test_data = generateRandomTestData(5)

print(f"\n Test Verileri ve Sonuçlar :")

# Random olarak üretilen test verisi kullanılarak tahminler üretilir. Her test verisi ve veriye ait tahmin değeri yazdırılır.
for test in test_data:
    output = xor_nn.predict(test.reshape(1, -1))  # Test verisini yeniden şekillendir
    print(f"Giriş: {test} -> Çıkış: {output[0][0]:.4f}")