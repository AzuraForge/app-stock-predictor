# AzuraForge: Hisse Senedi Tahmin Eklentisi

Bu proje, AzuraForge platformu için bir **uygulama eklentisidir**. `yfinance` kütüphanesinden aldığı geçmiş hisse senedi verilerini kullanarak, `azuraforge-learner` kütüphanesindeki `LSTM` modeli ile gelecekteki fiyat hareketlerini tahmin etmeye yönelik bir pipeline içerir.

Bu eklenti, hava durumu tahmin eklentisi ile birlikte, AzuraForge ekosistemine yeni bir uygulama eklerken takip edilmesi gereken standartları belirler.

## 🎯 Ana Sorumluluklar

*   Python `entry_points` mekanizması aracılığıyla kendisini AzuraForge ekosistemine bir "pipeline" olarak kaydeder.
*   Kullanıcı arayüzünde (`Dashboard`) dinamik olarak bir form oluşturulabilmesi için gerekli konfigürasyon ve şema dosyalarını sağlar.
*   `TimeSeriesPipeline` soyut sınıfını miras alarak veri çekme, ön işleme (veri temizleme dahil), özellik mühendisliği (OHLCV kullanımı) ve eğitim adımlarını uygular.

---

## 🏛️ Ekosistemdeki Yeri

Bu eklenti, AzuraForge ekosisteminin modüler ve genişletilebilir yapısının canlı bir örneğidir. Projenin genel mimarisini, vizyonunu ve geliştirme rehberini anlamak için lütfen ana **[AzuraForge Platform Dokümantasyonuna](https://github.com/AzuraForge/platform/tree/main/docs)** başvurun.

---

## 🛠️ İzole Geliştirme ve Hızlı Test

Bu eklenti, tüm AzuraForge platformunu (`Docker`) çalıştırmadan, tamamen bağımsız olarak test edilebilir.

### Gereksinimler
1.  Ana platformun **[Geliştirme Rehberi](https://github.com/AzuraForge/platform/blob/main/docs/DEVELOPMENT_GUIDE.md)**'ne göre Python sanal ortamınızın kurulu ve aktif olduğundan emin olun.
2.  Bu reponun kök dizininde olduğunuzundan emin olun.

### Testi Çalıştırma (CPU üzerinde)
Aşağıdaki komut, pipeline'ı `MSFT` için varsayılan ayarlarla çalıştıracaktır:
```bash
python tools/run_isolated.py
```

### Parametreleri Değiştirerek Test Etme
Farklı bir hisse senedi ve daha az epoch ile denemek için:
```bash
python tools/run_isolated.py --ticker AAPL --epochs 20
```

### Testi Çalıştırma (GPU üzerinde)
Eğer sisteminizde uyumlu bir NVIDIA GPU ve CUDA kurulu ise, eğitimi GPU üzerinde çalıştırmak için `AZURAFORGE_DEVICE` ortam değişkenini ayarlayabilirsiniz. Bu, eğitim sürecini önemli ölçüde hızlandıracaktır.

**Windows (PowerShell):**
```powershell
$env:AZURAFORGE_DEVICE="gpu"; python tools/run_isolated.py
```

**Linux / macOS / WSL:**
```bash
AZURAFORGE_DEVICE=gpu python tools/run_isolated.py
```
