import numpy as np

# Sigmoid aktivasyon fonksiyonu
# Gizli katman ve çıktı katmanının çıktılarını üreten fonksiyondur.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid aktivasyon fonksiyonunun türevi
def sigmoidDerivative(x):
    return x * (1 - x)

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

        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)

        return self.output_layer_output

    # Geri besleme fonksiyonu
    # YSA modelini eğitme aşamasında kullanılır.
    # Eğitim veri setlerinde gerçek çıktı değerleri bellidir.
    # Bu fonksiyon kullanılarak ileri besleme fonksiyonunda üretilen çıktılar ile eğitim veri setindeki beklenen çıktılar kıyaslanır.
    # Beklenen çıktı ile alınan çıktının arasındaki fark 0'a yakın olmalıdır. Modelimizin öğrenme aşaması bu fonksiyondur.
    # Beklenen çıktı ile alınan çıktı arasındaki farkı 0'a yaklaştıracak şekilde ağırlık vektörlerini ve bias'i günceller.
    # Modelimizin öğrenme aşaması burada gerçekleşir diyebiliriz.
    def backwardFeeding(self, X, y, learning_rate=0.1):

        error_output_layer = y - self.output_layer_output
        delta_output_layer = error_output_layer * sigmoidDerivative(self.output_layer_output)

        error_hidden_layer = delta_output_layer.dot(self.weights_hidden_output.T)
        delta_hidden_layer = error_hidden_layer * sigmoidDerivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(delta_output_layer) * learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden_layer) * learning_rate
        self.bias_output += np.sum(delta_output_layer, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate

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
