# Oyun Satış Tahmini

Bir oyun piyasaya çıkmadan "bu tutar mı?" tahmini yapabilmek yayıncıların rüyası. Biz de geçmiş satış verisine bakıp bir oyunun ne kadar satacağını kestirmeye çalışacağız.

## Veri seti

[Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (Kaggle, `gregorut/videogamesales`) kullanıldı: 1980-2020 arası 16.598 oyun, platform, tür, yayıncı, çıkış yılı ve bölgesel (Kuzey Amerika, Avrupa, Japonya, Diğer) + küresel satış rakamlarıyla birlikte. Hedef değişken `Global_Sales` (milyon adet, sürekli sayı) olduğu için bu bir **regresyon** problemi.

`Year` sütununda 271 satır (%1,63), `Publisher` sütununda 58 satır (%0,35) eksik. `Year`, aynı platformun medyan çıkış yılıyla; `Publisher`, veri setinde zaten var olan `"Unknown"` kategorisiyle dolduruldu. 578 farklı yayıncı tek tek one-hot kodlanmak yerine en sık geçen 30 yayıncı ayrı tutulup geri kalanı `"Diğer"` grubunda toplandı (oyunların %26,5'i bu gruba düşüyor).

### Veri sızıntısı tespiti — modele girmeyen sütunlar

Veri setinde `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales` ve `Rank` sütunları da var, ama bunlar özellik olarak kullanılmadı. Sebebi basit bir kontrolle ortaya çıktı: dört bölgesel satışın toplamı, yuvarlama farkı (en fazla 0,02 milyon) dışında `Global_Sales`'in tam kendisi. `Rank` ise `Global_Sales`'in azalan sıralamasından türetilmiş (Spearman korelasyonu -0,9996). Bu beş sütunu modele vermek, hedefi neredeyse bir toplama işlemiyle geri vermek anlamına gelirdi ve gerçek olmayan, yapay olarak mükemmel bir sonuç üretirdi. Bu yüzden modelleme sadece piyasaya çıkmadan önce zaten bilinen bilgilerle yapıldı: **Platform, Tür, Yayıncı, Yıl**. Bölgesel satış sütunları yalnızca betimleyici (EDA) görsellerde, modeli beslemeden kullanıldı.

## Hedef dönüşümü — neden `log1p(Global_Sales)`

`Global_Sales` ciddi şekilde sağa çarpık (çarpıklık katsayısı 17,4): medyan 0,17 milyon, ortalama 0,54 milyon, maksimum 82,74 milyon (Wii Sports). Bunu havada bırakmamak için aynı Random Forest'ı hem ham hedefte hem `log1p(Global_Sales)` üzerinde eğitip test RMSE'sini karşılaştırdık: ham hedefte 2,0223, log dönüşümünde (geri dönüştürülmüş) 2,0017. Yaklaşık %1'lik bir iyileşme, yani tek başına büyük bir kanıt değil; dürüstçe söylemek gerekirse küçük bir fark. Kararı asıl destekleyen teorik gerekçe şu: `log1p` olmadan modelin optimize ettiği kare hata neredeyse tamamen birkaç blockbuster'ın hatasından ibaret kalıyor (82 milyonluk bir oyunda 5 milyonluk hata, 0,17 milyonluk bir oyundaki hatadan kat kat ağır basıyor). Küçük ama tutarlı ampirik kazanç ile bu teorik gerekçe bir araya gelince `log1p` kullanmaya karar verdik. Üç model de bu dönüştürülmüş hedefte eğitildi, tüm sonuçlar raporlanmadan önce `expm1` ile orijinal (milyon adet) ölçeğe geri çevrildi.

## Modelleme

Kategorik sütunlar (Platform, Tür, Yayıncı grubu) one-hot kodlandı, `Year` `StandardScaler` ile ölçeklendi. Veri, satış dağılımının çarpıklığını dengelemek için `log1p(Global_Sales)`'in 5 eşit-frekans dilimine göre tabakalı örneklemeyle %80/%20 ayrıldı: 13.278 eğitim, 3.320 test satırı.

Üç model, aynı `RandomizedSearchCV` düzeniyle (5 katlı çapraz doğrulama, 15-18 kombinasyon denemesi, skor negatif RMSE) ayarlandı: **Random Forest**, **SVR** (Destek Vektör Regresyonu) ve ek üçüncü model olarak **XGBoost**. SVR'nin eğitim maliyeti örneklem büyüklüğüyle hızla arttığı için hiper-parametre araması eğitim setinden alınan 4.000 satırlık bir alt örneklemde yapıldı; diğer iki model tam eğitim setini kullandı. Üçü de aynı, hiç dokunulmamış test setinde değerlendirildi.

## Sonuçlar

Test setinde (3.320 oyun), RMSE'ye göre sıralı:

| Model | RMSE (milyon) | MAE (milyon) | R² |
|---|---|---|---|
| XGBoost | 1,8921 | 0,4524 | 0,0971 |
| Random Forest | 1,9045 | 0,4577 | 0,0852 |
| SVR | 1,9527 | 0,4500 | 0,0383 |

En iyi model XGBoost, ama bunu süslemeden söylemek gerekiyor: R² sadece 0,097. Yani Platform, Tür, Yayıncı ve Yıl bilgisi, bir oyunun küresel satışındaki değişimin sadece onda birini açıklayabiliyor. Bu bir modelleme hatası değil, veri setinin doğal sınırı: bir oyunun gerçekte ne kadar satacağını belirleyen asıl etkenler (oyunun kalitesi, pazarlama bütçesi, eleştirmen puanları, viral etki) bu veri setinde hiç yok. Model "Nintendo'nun first-party bir oyunu ortalamadan daha çok satar" gibi kaba ama gerçek eğilimleri yakalıyor, ama belirli bir oyunun tam olarak kaç milyon satacağını kestiremiyor.

Özellik önemine bakınca bu netleşiyor: gruplara göre toplam önem Yayıncı %70, Platform %21, Tür %8, Yıl sadece %1. Yayıncı grubunun payının büyük kısmı da tek bir kategoriden geliyor: `Publisher_Grup_Nintendo` (0,33), ikinci sıradaki Electronic Arts'ın (0,041) sekiz katından fazla. Yani model, işin özünde büyük ölçüde "bu oyun Nintendo'nun mu" sorusuna indirgeniyor. Gerçek vs tahmin grafiğinde (`gorseller/05_gercek_vs_tahmin.png`) de aynı hikaye görülüyor: noktalar y=x çizgisi civarında toplansa da düşük satışlı oyunlarda dağılım geniş, yüksek satışlı birkaç blockbuster ise sistematik olarak hafif düşük tahmin ediliyor.

## Görseller (`gorseller/`)

1. `01_platform_satis_treemap.png` / `.html` — platforma göre toplam küresel satış (Plotly treemap). PS2 açık ara lider (~1.256 milyon adet), X360 ve PS3 onu izliyor.
2. `02_tur_bolge_isi_haritasi.png` / `.html` — tür x bölge satış dağılımı, her tür için bölge payı (%). Role-Playing türü Japonya'da diğer türlere göre belirgin daha yüksek pay alıyor (%38), bu JRPG kültürünün bilinen ağırlığıyla örtüşüyor.
3. `03_yillik_satis_trendi.png` / `.html` — yıllara göre bölge kırılımlı satış trendi (Plotly alan grafiği). Zirve 2009; sonrası düşüş, dijital dağıtımın yükselişiyle ve veri toplamanın 2016 sonrası neredeyse durmasıyla ilgili.
4. `04_en_cok_satan_10_yayinci.png` / `.html` — en çok satan 10 yayıncı (Plotly bar). Nintendo (1.786 milyon), Electronic Arts (1.110 milyon) ve Activision (727 milyon) ilk üç sırada.
5. `05_gercek_vs_tahmin.png` / `.html` — gerçek vs tahmin (en iyi model XGBoost), log-log eksende çünkü hedef 4 basamak yayılıyor (0,01-82,74 milyon).
6. `06_ozellik_onemi.png` / `.html` — ilk 15 özellik önemi, gruba göre renklendirilmiş (Yayıncı/Platform/Tür).

## Notebook

`proje.ipynb`, önce `proje.py` script olarak yazılıp test edildi, `jupytext` ile notebook'a çevrildi, sonra `jupyter nbconvert --execute` ile gerçekten baştan sona çalıştırıldı. 19 kod hücresinin tamamında çıktı üretildiği, hata olmadığı Python ile ayrıca doğrulandı.

## Belirsizlikler

1. Aynı oyun (örn. bir GTA V) birden fazla platformda ayrı satır olarak geçiyor; eğitim ve test setine aynı ismin farklı platform sürümleri dağılmış olabilir. Bu klasik anlamda bir sızıntı değil (Platform sütunu değişiyor) ama ilişkili örnekler test performansını gerçekte olduğundan hafif iyimser gösterebilir. Güven: LOW-MEDIUM. Bunu netleştirmek isteyen biri `Name` bazlı `GroupKFold` ile bölmeyi tekrarlayıp RMSE farkına bakabilir.
2. Yayıncı grubunda top-30 eşiği (kapsam %73,5) sabit bir tercih; 20 veya 50 gibi farklı bir eşik test edilmedi. Güven: MEDIUM, sonucu büyük ölçüde değiştirmesi beklenmiyor çünkü zaten en önemli tekil kategori (Nintendo) top-30 içinde.
3. `Year` doldurma stratejisi (platform medyanı) mantıklı bir varsayım ama gerçek çıkış yılının yerini tutmuyor; bu 271 satır (%1,6) toplam sonucu ölçülebilir şekilde etkileyecek kadar büyük değil.

## Kullanılan kütüphaneler

- [pandas](https://pandas.pydata.org/docs/) — veri işleme
- [NumPy](https://numpy.org/doc/) — sayısal işlemler
- [scikit-learn](https://scikit-learn.org/stable/) — ön işleme, Random Forest, SVR, `RandomizedSearchCV`, metrikler
- [XGBoost](https://xgboost.readthedocs.io/) — gradyan artırmalı ağaç regresyonu
- [Plotly](https://plotly.com/python/) — tüm interaktif grafikler (treemap, heatmap, alan grafiği, bar, scatter)
- [SciPy](https://docs.scipy.org/doc/scipy/) — `RandomizedSearchCV` parametre dağılımları (`loguniform`, `randint`, `uniform`)
- [Kaleido](https://github.com/plotly/Kaleido) — Plotly grafiklerini statik PNG olarak dışa aktarma
- [Jupytext](https://jupytext.readthedocs.io/) — script <-> notebook dönüşümü
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) — veri seti indirme

## Dosya yapısı

```
14_oyun_satisi/
├── proje.py                        # ana script (kaynak), jupytext ile notebook'a çevrildi
├── proje.ipynb                     # çalıştırılmış, çıktıları doğrulanmış notebook
├── model_karsilastirma_tablosu.csv # 3 modelin test seti metrikleri
├── requirements.txt
├── README.md
├── veri/
│   └── vgsales.csv
└── gorseller/
    ├── 01_platform_satis_treemap.png / .html
    ├── 02_tur_bolge_isi_haritasi.png / .html
    ├── 03_yillik_satis_trendi.png / .html
    ├── 04_en_cok_satan_10_yayinci.png / .html
    ├── 05_gercek_vs_tahmin.png / .html
    └── 06_ozellik_onemi.png / .html
```
