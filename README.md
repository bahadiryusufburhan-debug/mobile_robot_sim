# Mobil Robot Navigasyon ve Sensör Füzyonu Simülasyonu

Bu proje, bir mobil robotun engellerle dolu 2 boyutlu bir ortamda otonom navigasyonunu, lokalizasyonunu ve sensör füzyonunu (Extended Kalman Filter) simüle etmektedir.

## Gereksinimler
- Python 3.8 veya üzeri
- NumPy
- Matplotlib

## Kurulum
1. Repoyu klonlayın veya proje klasörüne gidin:
   ```bash
   git clone <repo-url>
   cd mobile_robot_sim
   ```
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy matplotlib
   ```

## Kullanım
Simülasyonu başlatmak ve sonuçları (grafikler ve hata metrikleri) görmek için ana dosyayı çalıştırın:
```bash
python3 main.py
```
Bu komut, robotun hedefe ulaşmasını simüle edecek ve işlem bitiminde `simulation_results.png` adında 6 alt grafikten oluşan görsel bir rapor oluşturacaktır. Terminalde ayrıca RMSE ve MAE hata analizi sonuçları yazdırılacaktır.

Animasyonlu GIF çıktısı (simülasyonu görsel olarak izlemek) almak için:
```bash
python3 animate.py
```

## Dosya Yapısı ve Mimari
- `main.py`: Ana simülasyon döngüsü ve matplotlib ile görselleştirme (Zorunlu 6 grafik).
- `animate.py`: Robotun yörüngesini adım adım çizen ve `simulation.gif` üreten script.
- `robot.py`: Non-holonomic (diferansiyel sürüş) kinematik robot modeli.
- `environment.py`: 20x20 ortam, engeller, başlangıç ve hedefin tanımlandığı sınıf.
- `sensors.py`: LiDAR, IMU ve Tekerlek Enkoderi için gürültü simülasyon modelleri.
- `kalman_filter.py`: Sensör füzyonu için Genişletilmiş Kalman Filtresi (Extended Kalman Filter - EKF).
- `localization.py`: Dead Reckoning lokalizasyonu ve RMSE/MAE hata hesaplama fonksiyonları.
- `path_planner.py`: A* (A-Star) tabanlı engelden kaçınan grid-based yol planlayıcı.
- `controller.py`: Pure-Pursuit nokta izleme (waypoint following) kontrol algoritması.
