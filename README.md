# AzuraForge: Hisse Senedi Tahmin Uygulaması

Bu proje, AzuraForge platformu için bir **uygulama eklentisidir**. `yfinance` kütüphanesinden aldığı geçmiş hisse senedi verilerini kullanarak, `azuraforge-learner` kütüphanesindeki `LSTM` modeli ile gelecekteki fiyat hareketlerini tahmin etmeye yönelik bir pipeline içerir.

## 🎯 Ana Sorumluluklar

*   Python `entry_points` mekanizması aracılığıyla kendisini AzuraForge ekosistemine bir "pipeline" olarak kaydeder.
*   Kullanıcı arayüzünde (`Dashboard`) dinamik olarak bir form oluşturulabilmesi için gerekli konfigürasyon (`stock_predictor_config.yml`) ve şema (`form_schema.json`) dosyalarını sağlar.
*   `TimeSeriesPipeline` soyut sınıfını miras alarak veri çekme, ön işleme, model oluşturma ve eğitim adımlarını kendi uzmanlık alanına göre (hisse senedi verisi) uygular.

---

## 🏛️ Ekosistemdeki Yeri

Bu eklenti, AzuraForge ekosisteminin modüler ve genişletilebilir yapısının canlı bir örneğidir. Projenin genel mimarisini, vizyonunu ve geliştirme rehberini anlamak için lütfen ana **[AzuraForge Platform Dokümantasyonuna](https://github.com/AzuraForge/platform/tree/main/docs)** başvurun.

---

## 🛠️ Kurulum ve Geliştirme

Bu eklenti, `worker` servisi tarafından bir bağımlılık olarak kurulur ve çalıştırılır. Yerel geliştirme ortamı kurulumu için ana platformun **[Geliştirme Rehberi](https://github.com/AzuraForge/platform/blob/main/docs/DEVELOPMENT_GUIDE.md)**'ni takip edin.