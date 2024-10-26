import numpy as np

# Sigmoid aktivasyon fonksiyonu
# Gizli katman ve çıktı katmanının çıktılarını üreten fonksiyondur.
def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    print("\nSigmoid fonksiyonu: f(x) = 1 / (1 + exp(-x))")
    print(f"Sigmoid giriş: {x}")
    print(f"Sigmoid çıkış: {result}")
    return result

# Sigmoid aktivasyon fonksiyonunun türevi
def sigmoidDerivative(x):
    result = x * (1 - x)
    print("\nSigmoid Türev fonksiyonu: f'(x) = x * (1 - x)")
    print(f"Sigmoid Türev giriş: {x}")
    print(f"Sigmoid Türev çıkış: {result}")
    return result

# Sinir ağı sınıfı
class XOR_NN:
    def __init__(self):

        # giriş katmanı nöron sayısı
        self.input_neurons = 2
        # gizli katman nöron sayısı
        self.hidden_neurons = 2
        # çıkış katmanı nöron sayısı
        self.output_neurons = 1

        # giriş katmanı ağırlık vektörü
        self.weights_input_hidden = np.random.uniform(size=(self.input_neurons, self.hidden_neurons))
        # gizli katman çıktısından sonra çıkış katmanında kullanılacak ağırlık vektörü
        self.weights_hidden_output = np.random.uniform(size=(self.hidden_neurons, self.output_neurons))

        # gizli katman bias değeri
        self.bias_hidden = np.random.uniform(size=(1, self.hidden_neurons))
        # çıkış katmanı bias değeri
        self.bias_output = np.random.uniform(size=(1, self.output_neurons))

        # Ağırlık ve bias'ları, kayıp fonksiyonunu her epoch için kaydetmek üzere listeler tanımlanır
        self.epoch_losses = []
        self.epoch_weights_input_hidden = []
        self.epoch_weights_hidden_output = []
        self.epoch_bias_hidden = []
        self.epoch_bias_output = []
        self.epoch_delta_output_layer = []
        self.epoch_delta_hidden_layer = []

    # İleri besleme fonksiyonu.
    # Burada amaç fonksiyona gelen X değerinin gizli ve çıktı katmanlarındaki ağıırlık ve bias değerleri ile matematiksel işlemlere tabi tutarak bir çıktı
    # üretmesini sağlamaktır.
    def forwardFeeding(self, X):
        print("\n\n--- İleri Besleme (Forward Feeding) ---")
        print("Gizli katman girdi hesaplaması: Z_h = X * W_ih + b_h")
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        print(f"Gizli Katman Girdi: {self.hidden_layer_input}")

        print("Sigmoid aktivasyon (gizli katman): H = sigmoid(Z_h)")
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        print(f"Gizli Katman Çıktı (Aktivasyon Sonrası): {self.hidden_layer_output}")

        print("Çıkış katmanı girdi hesaplaması: Z_o = H * W_ho + b_o")
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        print(f"Çıkış Katmanı Girdi: {self.output_layer_input}")

        print("Sigmoid aktivasyon (çıkış katmanı): Y = sigmoid(Z_o)")
        self.output_layer_output = sigmoid(self.output_layer_input)
        print(f"Çıkış Katmanı Çıktı (Aktivasyon Sonrası): {self.output_layer_output}")

        return self.output_layer_output

    # Geri besleme fonksiyonu
    # YSA modelini eğitme aşamasında kullanılır.
    # Eğitim veri setlerinde gerçek çıktı değerleri bellidir.
    # Bu fonksiyon kullanılarak ileri besleme fonksiyonunda üretilen çıktılar ile eğitim veri setindeki beklenen çıktılar kıyaslanır.
    # Beklenen çıktı ile alınan çıktının arasındaki fark 0'a yakın olmalıdır. Modelimizin öğrenme aşaması bu fonksiyondur.
    # Beklenen çıktı ile alınan çıktı arasındaki farkı 0'a yaklaştıracak şekilde ağırlık vektörlerini ve bias'i günceller.
    # Modelimizin öğrenme aşaması burada gerçekleşir diyebiliriz.
    def backwardFeeding(self, X, y, learning_rate=0.1):
        print("\n\n--- Geri Besleme (Backward Feeding) ---")

        print("Çıkış katmanı hatası: E_o = y - Y")
        error_output_layer = y - self.output_layer_output
        print(f"Çıkış Katmanı Hata: {error_output_layer}")

        print("Çıkış katmanı delta değeri: Δ_o = E_o * sigmoid'(Y)")
        delta_output_layer = error_output_layer * sigmoidDerivative(self.output_layer_output)
        print(f"Çıkış Katmanı Delta: {delta_output_layer}")

        print("Gizli katman hatası: E_h = Δ_o * W_ho^T")
        error_hidden_layer = delta_output_layer.dot(self.weights_hidden_output.T)
        print(f"Gizli Katman Hata: {error_hidden_layer}")

        print("Gizli katman delta değeri: Δ_h = E_h * sigmoid'(H)")
        delta_hidden_layer = error_hidden_layer * sigmoidDerivative(self.hidden_layer_output)
        print(f"Gizli Katman Delta: {delta_hidden_layer}")

        print("\nAğırlık ve bias güncellemeleri:")
        print("W_ho = W_ho + H^T * Δ_o * öğrenme hızı")
        self.weights_hidden_output += self.hidden_layer_output.T.dot(delta_output_layer) * learning_rate
        print(f"Güncellenmiş Gizli-Çıkış Ağırlıkları: {self.weights_hidden_output}")

        print("W_ih = W_ih + X^T * Δ_h * öğrenme hızı")
        self.weights_input_hidden += X.T.dot(delta_hidden_layer) * learning_rate
        print(f"Güncellenmiş Girdi-Gizli Ağırlıkları: {self.weights_input_hidden}")

        print("b_o = b_o + Σ(Δ_o) * öğrenme hızı")
        self.bias_output += np.sum(delta_output_layer, axis=0, keepdims=True) * learning_rate
        print(f"Güncellenmiş Çıkış Bias: {self.bias_output}")

        print("b_h = b_h + Σ(Δ_h) * öğrenme hızı")
        self.bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate
        print(f"Güncellenmiş Gizli Bias: {self.bias_hidden}")

        # Delta değerleri kaydedilir
        self.epoch_delta_output_layer.append(delta_output_layer)
        self.epoch_delta_hidden_layer.append(delta_hidden_layer)


    # Eğitim veri seti ile ysa modelinin eğitimini yapan fonksiyon.
    # iterations parametresi eğitim veri setinin kaç kez kullanılarak modelin ağırlık ve bias değerlerini güncelleyeceğini belirtir. Burada default olarak
    # 10000 kez kullanması istenmiş.
    # X parametresi XOR problemine ait eğitim veri setidir.
    # y parametresi X eğitim veri setinin gerçek çıktılarıdır.
    # learningRate modelin eğitim hızıdır.
    # Her döngüde (iterasyonda) forwardFeeding (ileri besleme) fonksiyonu X veri seti parametresi verilerek çalıştırılır ve çıktılar hesaplanır.
    # backwardFeeding (geri besleme) fonksiyonu X veri seti, veri setinin beklenen çıktıları ve sabit tutulan öğrenme hızı ile çağrılır. Burada beklenen çıktı
    # ile forwardFeeding'in hesapladığı çıktı kıyaslanarak ağırlıklar ve bias değerleri güncellenir. Modelin öğrenme aşaması bu kısımdır.
    # Belirli sayıda döngüde loss (hatalı hesaplama miktarı) hesaplanarak yazdırılır. Döngü ilerledikçe bu değerin de sıfıra yaklaşması beklenir.
    def train(self, X, y, iterations=10000, learningRate=0.1):
        for iteration in range(iterations):
            self.forwardFeeding(X)
            self.backwardFeeding(X, y, learningRate)

            # Her epoch sonunda loss kaydedilir
            loss = np.mean(np.square(y - self.output_layer_output))
            self.epoch_losses.append(loss)

            # Her epoch sonunda ağırlık ve bias değerleri kaydedilir
            self.epoch_weights_input_hidden.append(self.weights_input_hidden.copy())
            self.epoch_weights_hidden_output.append(self.weights_hidden_output.copy())
            self.epoch_bias_hidden.append(self.bias_hidden.copy())
            self.epoch_bias_output.append(self.bias_output.copy())

            # Her 1000 iterasyonda 1 kaybı yazdır
            if iteration % 1000 == 0:
                loss = np.mean(np.square(y - self.output_layer_output))
                print(f"Döngü {iteration}/{iterations}, Loss: {loss}")

    # Tahmin yapma fonksiyonu
    # Eğitim veri seti ile ysa modeli eğitildikten sonra test veri seti ile eğitilen ysa modelini test ettiğimiz fonksiyondur.
    # burada yalnızca forwardFeeding (ileri besleme) fonksiyonu kullanılır. Eğitim veri seti ile eğitilen modelin, çıktıları bilinmen verilere üreteceği sonuç
    # hesaplanır.
    def predict(self, X):
        return self.forwardFeeding(X)
