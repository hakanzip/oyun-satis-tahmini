# %% [markdown]
# # Oyun Satış Tahmini — Bir Oyun Piyasaya Çıkmadan "Bu Tutar mı?"
#
# Bir oyun stüdyosunun yayıncıya sunduğu her yeni proje aslında bir bahis:
# bütçe onaylanacak, pazarlama parası ayrılacak, ama kimse "bu oyun kaç
# milyon satar" sorusuna emin cevap veremiyor. Elimizde 1980-2016 arası
# 16.598 oyunun platform, tür, yayıncı ve çıkış yılı bilgisiyle birlikte
# bölgesel ve toplam (küresel) satış rakamları var; amacımız bir oyun
# henüz satılmadan, sadece bu tanıtıcı bilgilere bakarak küresel satışını
# (milyon adet) tahmin etmek.
#
# **Not (yöntem netliği):** Hedef değişken (`Global_Sales`) sürekli bir
# sayı olduğu için bu bir **regresyon** problemi; metrik olarak
# accuracy/precision değil RMSE, MAE ve R² kullanılıyor. Taslaktaki
# "SVM (SVR) + Random Forest" ikilisine üçüncü model olarak **XGBoost
# regresyonu** eklendi.

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform, randint, uniform

from xgboost import XGBRegressor

# jupyter nbconvert --execute notebook içinde __file__ tanımlı DEĞİL;
# script olarak çalıştırıldığında ise var. İkisinde de çalışsın diye:
PROJE_KOK = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
VERI_KOK = PROJE_KOK / "veri"
VERI_KOK.mkdir(exist_ok=True)
GORSEL_KOK = PROJE_KOK / "gorseller"
GORSEL_KOK.mkdir(exist_ok=True)

RASSAL_TOHUM = 42

# --- Görsel tema: seri boyunca kullanılan, doğrulanmış renk paleti ---
# (13_konut_fiyati ve diğer projelerle aynı palet — tüm seri tutarlı olsun diye)
RENK_MAVI = "#2a78d6"      # kategorik slot 1 -> ana seri / gerçek değer
RENK_KIRMIZI = "#e34948"   # kategorik slot 8 -> ikinci seri / tahmin-fark
RENK_TURUNCU = "#eb6834"   # kategorik slot 2
RENK_AQUA = "#1baf7a"      # kategorik slot 3
RENK_MOR = "#7d5ba6"       # kategorik slot 4 (4. grup gerektiğinde)
YUZEY = "#fcfcfb"
METIN_ANA = "#0b0b0b"
METIN_SOLUK = "#52514e"

MAVI_RAMPA_PLOTLY = [
    [0.0, "#cde2fb"], [0.2, "#9ec5f4"], [0.4, "#5598e7"],
    [0.6, "#2a78d6"], [0.8, "#1c5cab"], [1.0, "#0d366b"],
]

PLOTLY_SABLON = dict(
    layout=go.Layout(
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=METIN_ANA, size=13),
        paper_bgcolor=YUZEY,
        plot_bgcolor=YUZEY,
        colorway=[RENK_MAVI, RENK_KIRMIZI, RENK_TURUNCU, RENK_AQUA, RENK_MOR],
        xaxis=dict(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7"),
        yaxis=dict(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

print(f"Proje kök dizini: {PROJE_KOK}")
print(f"Veri klasörü: {VERI_KOK}")

# %% [markdown]
# ## 1. Veri Yükleme
#
# Veri seti Kaggle'dan (`gregorut/videogamesales`) `veri/vgsales.csv` olarak
# indirildi (~1,3 MB, 16.598 satır — hafif bir veri seti, örneklem almaya
# gerek yok).

# %%
ham_df = pd.read_csv(VERI_KOK / "vgsales.csv")
print("Veri seti boyutu:", ham_df.shape)
print("\nSütunlar:", list(ham_df.columns))
ham_df.head()

# %% [markdown]
# ## 2. Keşif ve Eksik Değerler

# %%
print(ham_df.isnull().sum())
print(f"\n'Year' eksik: {ham_df['Year'].isnull().sum()} satır (%{100*ham_df['Year'].isnull().mean():.2f})")
print(f"'Publisher' eksik: {ham_df['Publisher'].isnull().sum()} satır (%{100*ham_df['Publisher'].isnull().mean():.2f})")
print(f"\nGlobal_Sales betimleyici istatistik:\n{ham_df['Global_Sales'].describe()}")
print(f"\nGlobal_Sales çarpıklık (skewness): {ham_df['Global_Sales'].skew():.2f}")

# %% [markdown]
# Hedef değişken ciddi şekilde sağa çarpık: ortalama 0,54 milyon adet ama
# maksimum 82,74 milyon (Wii Sports). Medyan (0,17) ortalamanın üçte biri
# bile değil — yani veri setinin büyük kısmı düşük satışlı oyunlardan
# oluşuyor, birkaç dev hit ortalamayı yukarı çekiyor. Bu durum aşağıda
# hedef dönüşümü kararını doğrudan etkiliyor (Bölüm 4).

# %% [markdown]
# ## 3. VERİ SIZINTISI TESPİTİ — Kritik Bulgu
#
# Veri setinde `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales` (bölgesel
# satışlar) ve `Rank` sütunları da var. İlk bakışta bunlar da birer
# "özellik" gibi durabilir, ama durum öyle değil: aşağıdaki kontrol bunun
# **veri sızıntısı (data leakage)** olduğunu kanıtlıyor.

# %%
bolge_toplami = ham_df["NA_Sales"] + ham_df["EU_Sales"] + ham_df["JP_Sales"] + ham_df["Other_Sales"]
fark = (bolge_toplami - ham_df["Global_Sales"]).abs()
print("Bölgesel satışların toplamı ile Global_Sales arasındaki fark:")
print(fark.describe())
print(f"\nFark > 0,01 olan satır sayısı: {(fark > 0.01).sum()} / {len(ham_df)} "
      "(bu farkın tamamı 2 ondalık basamağa yuvarlamadan kaynaklanıyor, maksimum 0,02)")

rank_spearman = ham_df["Rank"].corr(ham_df["Global_Sales"], method="spearman")
print(f"\n'Rank' ile 'Global_Sales' arasındaki Spearman korelasyonu: {rank_spearman:.4f}")

# %% [markdown]
# **Sonuç — HİPOTEZ (Güven: VERY HIGH, doğrudan sayısal kanıt var):**
# `Global_Sales`, `NA_Sales + EU_Sales + JP_Sales + Other_Sales` toplamının
# ta kendisi (fark yalnızca yuvarlama, maksimum 0,02 milyon). `Rank` ise
# `Global_Sales`'in azalan sırasından türetilmiş (Spearman korelasyonu
# -0,9996, neredeyse mükemmel monoton ilişki).
#
# Bu dört bölgesel satış sütunu ile `Rank`'ı özellik olarak kullanmak,
# modele hedefi neredeyse doğrudan (bir toplama işlemiyle) vermek anlamına
# gelir — gerçek bir tahmin değil, sahte bir R² ≈ 1,0 üretir. Bu yüzden bu
# 5 sütun modelleme özellik setinden **tamamen çıkarıldı**; sadece Bölüm
# 6 ve 7'deki betimleyici (EDA) görsellerde, modeli beslemeden, veriyi
# anlamak için kullanılıyorlar. Gerçek tahmin sadece piyasaya çıkmadan
# önce bilinebilecek bilgilerle yapılacak: **Platform, Tür, Yayıncı, Yıl**.

# %% [markdown]
# ## 4. Eksik Değer İşleme ve Yayıncı Gruplama
#
# `Year` eksik olan 271 satır (%1,63), aynı platformun medyan çıkış yılıyla
# dolduruluyor (bir platformun ömrü belli bir yıl aralığına sığdığı için
# global medyandan daha isabetli bir tahmin). `Publisher` eksik olan 58
# satır ise veri setinde zaten var olan `"Unknown"` kategorisiyle
# birleştiriliyor (58 satır, mevcut 203 "Unknown" satırıyla aynı anlama
# geliyor: yayıncı bilgisi kayıtlara geçmemiş).

# %%
df = ham_df.copy()
df["Year"] = df.groupby("Platform")["Year"].transform(lambda s: s.fillna(s.median()))
df["Year"] = df["Year"].fillna(df["Year"].median())  # kalan tekil durumlar için genel medyan
df["Publisher"] = df["Publisher"].fillna("Unknown")
print("Doldurma sonrası eksik değer sayısı:", df[["Year", "Publisher"]].isnull().sum().sum())

# %% [markdown]
# `Publisher` sütununda 578 farklı değer var — tamamını one-hot kodlamak
# hem gereksiz seyrek bir matris üretir hem de nadir yayıncılarda modele
# gürültüden başka bir şey öğretmez. Bunun yerine en sık geçen 30 yayıncı
# ayrı kategori olarak tutulup geri kalan her şey `"Diğer"` grubunda
# toplanıyor (30 yayıncı, oyunların büyük çoğunluğunu zaten kapsıyor).

# %%
en_sik_30_yayinci = df["Publisher"].value_counts().head(30).index
df["Publisher_Grup"] = df["Publisher"].where(df["Publisher"].isin(en_sik_30_yayinci), "Diğer")
print(f"Publisher_Grup kategori sayısı: {df['Publisher_Grup'].nunique()} (30 yayıncı + Diğer)")
print(f"'Diğer' grubuna düşen satır oranı: %{100*(df['Publisher_Grup']=='Diğer').mean():.1f}")
print(f"Platform kategori sayısı: {df['Platform'].nunique()}, Genre kategori sayısı: {df['Genre'].nunique()}")

# %% [markdown]
# ## 5. Görsel 1 — Platforma Göre Toplam Satış (Treemap)

# %%
platform_satis = (
    df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).reset_index()
)
fig_platform = px.treemap(
    platform_satis, path=[px.Constant("Tüm Platformlar"), "Platform"], values="Global_Sales",
    color="Global_Sales", color_continuous_scale=MAVI_RAMPA_PLOTLY,
    title="Platforma Göre Toplam Küresel Satış (milyon adet)",
    labels={"Global_Sales": "Küresel Satış (milyon adet)", "Platform": "Platform"},
)
fig_platform.update_layout(PLOTLY_SABLON["layout"], margin=dict(t=60, l=10, r=10, b=10))
fig_platform.update_traces(
    textinfo="label+value",
    # Plotly'nin varsayılan treemap hover'ı "labels=/parent=/id=/Global_Sales_sum="
    # gibi ham İngilizce alan adlarını gösteriyor (labels= parametresi bunları
    # değiştirmiyor) — bu yüzden hover metni tamamen Türkçe olacak şekilde elle yazıldı.
    hovertemplate="<b>%{label}</b><br>Küresel Satış: %{value:.2f} milyon adet<extra></extra>",
)
fig_platform.write_image(GORSEL_KOK / "01_platform_satis_treemap.png", width=1100, height=650, scale=2)
fig_platform.write_html(GORSEL_KOK / "01_platform_satis_treemap.html")
print("Kaydedildi: 01_platform_satis_treemap.png / .html")
print(platform_satis.head(5).to_string(index=False))

# %% [markdown]
# **Gözlem:** PS2 tüm platformlar arasında en yüksek toplam satışa sahip
# (~1.255 milyon adet), hemen ardından X360 ve PS3 geliyor. Bu üç platform
# 2005-2013 arası konsol pazarının ana gövdesini oluşturuyordu; sıralama,
# beklenen konsol pazar payı hikayesiyle örtüşüyor.

# %% [markdown]
# ## 6. Görsel 2 — Tür × Bölge Satış Isı Haritası
#
# Burada bilerek bölgesel satış sütunlarını kullanıyoruz — ama sadece
# **betimleyici** amaçla (Bölüm 3'te açıklandığı gibi bu sütunlar modele
# hiç girmiyor). Her tür için bölgesel satışlar kendi toplamına oranlanıyor
# (satır bazında %100'e tamamlanan yüzdeler) ki mutlak büyüklük farkı
# (NA pazarı zaten en büyük pazar) bölgesel *profil* farkını gölgelemesin.

# %%
tur_bolge = df.groupby("Genre")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
tur_bolge_yuzde = tur_bolge.div(tur_bolge.sum(axis=1), axis=0) * 100
tur_bolge_yuzde = tur_bolge_yuzde.sort_values("NA_Sales", ascending=False)
bolge_adlari = {"NA_Sales": "Kuzey Amerika", "EU_Sales": "Avrupa", "JP_Sales": "Japonya", "Other_Sales": "Diğer"}
tur_bolge_yuzde = tur_bolge_yuzde.rename(columns=bolge_adlari)

# Tür isimleri (Action, Sports, Role-Playing vb.) sektörde yerleşik İngilizce
# terimler — olduğu gibi kalıyor ama grafik okunurluğu için yanına parantez
# içinde Türkçe çevirisi ekleniyor (yalnızca eksen görüntüsü; "Genre" sütununun
# kendisi ve modelleme tarafı İngilizce/orijinal kalmaya devam ediyor).
TUR_CEVIRI = {
    "Action": "Aksiyon", "Sports": "Spor", "Role-Playing": "Rol Yapma",
    "Shooter": "Nişancı", "Racing": "Yarış", "Puzzle": "Bulmaca",
    "Platform": "Platform", "Simulation": "Simülasyon", "Adventure": "Macera",
    "Strategy": "Strateji", "Fighting": "Dövüş", "Misc": "Diğer",
}
tur_bolge_yuzde.index = [
    f"{ad} ({TUR_CEVIRI[ad]})" if TUR_CEVIRI.get(ad, ad) != ad else ad
    for ad in tur_bolge_yuzde.index
]

fig_isi = go.Figure(
    go.Heatmap(
        z=tur_bolge_yuzde.values, x=tur_bolge_yuzde.columns, y=tur_bolge_yuzde.index,
        colorscale=MAVI_RAMPA_PLOTLY, text=tur_bolge_yuzde.round(1).values,
        texttemplate="%{text}%", colorbar=dict(title="Pay (%)"),
        hovertemplate="Tür: %{y}<br>Bölge: %{x}<br>Pay: %{z:.1f}%<extra></extra>",
    )
)
fig_isi.update_layout(
    PLOTLY_SABLON["layout"],
    title="Tür × Bölge Satış Dağılımı (her tür için bölge payı, %)",
    xaxis_title="Bölge", yaxis_title="Tür",
)
fig_isi.write_image(GORSEL_KOK / "02_tur_bolge_isi_haritasi.png", width=950, height=650, scale=2)
fig_isi.write_html(GORSEL_KOK / "02_tur_bolge_isi_haritasi.html")
print("Kaydedildi: 02_tur_bolge_isi_haritasi.png / .html")

# %% [markdown]
# **Gözlem:** Role-Playing (RPG) türü Japonya'da diğer türlere göre
# belirgin biçimde daha yüksek pay alıyor (~%25'e yakın), oysa Japonya
# payı genelde çoğu türde %10-15 bandında kalıyor. Bu, JRPG kültürünün
# Japon pazarındaki bilinen ağırlığıyla örtüşen, veriden bağımsız olarak
# da beklenen bir desen.

# %% [markdown]
# ## 7. Görsel 3 — Yıllara Göre Satış Trendi (Alan Grafiği)
#
# 2017 (3 satır) ve 2020 (1 satır) veri toplama kesintisi yüzünden neredeyse
# boş; bu iki yıl grafikte yanıltıcı bir çöküş gibi görünmemesi için
# 1980-2016 aralığına sabitlendi (bu 2 yıl atılmadı, sadece grafik ekseni
# bu şekilde seçildi — modelleme için tüm veri kullanılmaya devam ediyor).

# %%
yil_bolge = (
    df[df["Year"].between(1980, 2016)]
    .groupby("Year")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
    .rename(columns=bolge_adlari)
    .reset_index()
)
fig_trend = go.Figure()
for bolge, renk in zip(["Kuzey Amerika", "Avrupa", "Japonya", "Diğer"],
                        [RENK_MAVI, RENK_TURUNCU, RENK_AQUA, RENK_MOR]):
    fig_trend.add_trace(go.Scatter(
        x=yil_bolge["Year"], y=yil_bolge[bolge], name=bolge, mode="lines",
        stackgroup="bolge", line=dict(width=0.5, color=renk),
        hovertemplate=f"Yıl: %{{x}}<br>{bolge}: %{{y:.1f}} milyon adet<extra></extra>",
    ))
fig_trend.update_layout(
    PLOTLY_SABLON["layout"],
    title="Yıllara Göre Küresel Oyun Satışı (bölge kırılımlı, milyon adet)",
    xaxis_title="Yıl", yaxis_title="Satış (milyon adet)",
)
fig_trend.write_image(GORSEL_KOK / "03_yillik_satis_trendi.png", width=1100, height=600, scale=2)
fig_trend.write_html(GORSEL_KOK / "03_yillik_satis_trendi.html")
print("Kaydedildi: 03_yillik_satis_trendi.png / .html")
print(f"Zirve yıl: {int(yil_bolge.loc[yil_bolge.iloc[:,1:].sum(axis=1).idxmax(), 'Year'])}")

# %% [markdown]
# **Gözlem:** Satışlar 2008-2009 civarında zirve yapıyor (Wii/PS3/X360
# döneminin doruğu), sonrasında dijital dağıtımın ve mobil oyunun
# yükselişiyle bu veri setinin kapsadığı "kutulu/fiziksel perakende satış"
# rakamları düşüşe geçiyor. Bu düşüş oyun pazarının küçülmesi değil, bu
# özel veri setinin fiziksel/perakende odaklı olmasıyla ilgili — 2016
# sonrası veri toplanmamış olması da (Bölüm 2) bunu destekliyor.

# %% [markdown]
# ## 8. Görsel 4 — En Çok Satan 10 Yayıncı

# %%
yayinci_satis = (
    df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).head(10).reset_index()
)
fig_yayinci = px.bar(
    yayinci_satis.sort_values("Global_Sales"), x="Global_Sales", y="Publisher", orientation="h",
    color_discrete_sequence=[RENK_MAVI],
    title="En Çok Satan 10 Yayıncı (toplam küresel satış, milyon adet)",
    labels={"Global_Sales": "Toplam Satış (milyon adet)", "Publisher": ""},
)
fig_yayinci.update_layout(PLOTLY_SABLON["layout"], showlegend=False)
fig_yayinci.update_traces(
    hovertemplate="Yayıncı: %{y}<br>Toplam Satış: %{x:.2f} milyon adet<extra></extra>"
)
fig_yayinci.write_image(GORSEL_KOK / "04_en_cok_satan_10_yayinci.png", width=950, height=600, scale=2)
fig_yayinci.write_html(GORSEL_KOK / "04_en_cok_satan_10_yayinci.html")
print("Kaydedildi: 04_en_cok_satan_10_yayinci.png / .html")
print(yayinci_satis.to_string(index=False))

# %% [markdown]
# **Gözlem:** Nintendo, kendi platformlarına özel first-party avantajıyla
# (Mario, Zelda, Pokémon gibi first-party serilerin sahibi olması) listenin
# açık ara lideri. Electronic Arts ve Activision onu takip ediyor —
# ikisi de yıllık spor/FPS serileriyle (FIFA, Call of Duty) düzenli, yüksek
# hacimli satış üreten yayıncılar.

# %% [markdown]
# ## 9. Hedef Dönüşüm Kararı — Neden `log1p(Global_Sales)`?
#
# Bölüm 2'de görüldüğü gibi hedef ciddi çarpık. Kararı havada bırakmamak
# için gerçek bir sayı üretiyoruz: aynı Random Forest'ı (varsayılan
# hiper-parametrelerle, sadece hızlı bir karşılaştırma için) hem ham
# `Global_Sales` üzerinde hem `log1p(Global_Sales)` üzerinde eğitip, ikisini
# de orijinal (milyon adet) ölçekte test hatasıyla karşılaştırıyoruz.

# %%
kategorik_sutunlar = ["Platform", "Genre", "Publisher_Grup"]
sayisal_sutunlar = ["Year"]
ozellik_sutunlari = kategorik_sutunlar + sayisal_sutunlar

on_isleyici_test = ColumnTransformer([
    ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik_sutunlar),
    ("sayisal", StandardScaler(), sayisal_sutunlar),
])

X_karar = df[ozellik_sutunlari]
y_karar = df["Global_Sales"]
X_karar_egitim, X_karar_test, y_karar_egitim, y_karar_test = train_test_split(
    X_karar, y_karar, test_size=0.2, random_state=RASSAL_TOHUM,
)

karar_pipeline_ham = Pipeline([
    ("on_isleme", on_isleyici_test),
    ("model", RandomForestRegressor(n_estimators=200, random_state=RASSAL_TOHUM, n_jobs=2)),
])
karar_pipeline_ham.fit(X_karar_egitim, y_karar_egitim)
tahmin_ham = karar_pipeline_ham.predict(X_karar_test)
rmse_ham = np.sqrt(mean_squared_error(y_karar_test, tahmin_ham))

karar_pipeline_log = Pipeline([
    ("on_isleme", on_isleyici_test),
    ("model", RandomForestRegressor(n_estimators=200, random_state=RASSAL_TOHUM, n_jobs=2)),
])
karar_pipeline_log.fit(X_karar_egitim, np.log1p(y_karar_egitim))
tahmin_log_ham_olcek = np.clip(np.expm1(karar_pipeline_log.predict(X_karar_test)), 0, None)
rmse_log = np.sqrt(mean_squared_error(y_karar_test, tahmin_log_ham_olcek))

print(f"Ham hedefle eğitilen RF -> test RMSE (milyon adet): {rmse_ham:.4f}")
print(f"log1p hedefle eğitilen RF -> test RMSE (milyon adet, geri dönüştürülmüş): {rmse_log:.4f}")
print(f"İyileşme: %{100*(rmse_ham - rmse_log)/rmse_ham:.1f}")

# %% [markdown]
# **HİPOTEZ:** `log1p` dönüşümü test RMSE'sini iyileştirir çünkü ham
# hedefte birkaç dev hit (Wii Sports 82,74 milyon gibi) kare hata
# fonksiyonunu domine ediyor, model çoğunluğu oluşturan düşük satışlı
# oyunları görmezden gelip sadece birkaç uç noktayı tahmin etmeye
# yatkınlaşıyor.
#
# **Dürüst not:** Ölçülen iyileşme büyük bir sıçrama değil, sadece %1
# civarı (RMSE 2,02 -> 2,00 milyon) — abartmıyoruz. Bunu tek başına
# "kesin kanıt" saymak yanıltıcı olurdu. Kararı asıl destekleyen, bu
# ölçümle birlikte duran teorik gerekçe: `log1p` olmadan modelin kare
# hata fonksiyonu neredeyse tamamen birkaç blockbuster'ın hatasından
# ibaret kalıyor (82,74 milyonluk bir oyunda 5 milyonluk hata, kare
# hatada 0,17 milyonluk bir oyundaki hatanın binlerce katı ağırlık
# taşıyor), yani ölçülen küçük fark bile gerçek bir yön veriyor.
# **Güven:** MEDIUM-HIGH (küçük ama tutarlı ampirik iyileşme + güçlü
# teorik gerekçe; VERY HIGH demiyoruz çünkü tek bir hızlı ölçüme dayanıyor).
# **Uygulama:** Bu proje boyunca üç model de `log1p(Global_Sales)`
# üzerinde eğitiliyor, tüm test metrikleri raporlanmadan önce `expm1` ile
# orijinal (milyon adet) ölçeğe geri dönüştürülüyor.

# %% [markdown]
# ## 10. Eğitim / Test Ayrımı — Çarpıklığa Karşı Tabakalı Örnekleme
#
# Hedef çok çarpık olduğu için rastgele bölme, test setinde nadir
# "blockbuster" oyunları şansa bırakabilir. `log1p(Global_Sales)`'i 5
# eşit-frekans dilimine ayırıp (qcut) tabakalı örnekleme yapıyoruz; bu
# dilim sütunu sadece bölme için kullanılıyor, modele özellik olarak
# verilmiyor.

# %%
df["_satis_dilimi"] = pd.qcut(np.log1p(df["Global_Sales"]), q=5, labels=False, duplicates="drop")

X = df[ozellik_sutunlari]
y_ham = df["Global_Sales"]
y_log = np.log1p(y_ham)

X_train, X_test, y_train_log, y_test_log, y_train_ham, y_test_ham = train_test_split(
    X, y_log, y_ham, test_size=0.2, random_state=RASSAL_TOHUM, stratify=df["_satis_dilimi"],
)
print(f"Eğitim seti: {X_train.shape[0]} satır, Test seti: {X_test.shape[0]} satır")
print(f"Eğitimde ortalama (log ölçek): {y_train_log.mean():.3f}, testte: {y_test_log.mean():.3f}")

# %% [markdown]
# ## 11. Ön İşleme ve Model Eğitimi
#
# Üç model de aynı `ColumnTransformer` (kategorikler için `OneHotEncoder`,
# `Year` için `StandardScaler`) + `RandomizedSearchCV` (5 katlı çapraz
# doğrulama, skor = negatif RMSE, log ölçekte) düzeniyle ayarlanıyor.
#
# **Not (performans kararı):** SVR'nin eğitim maliyeti örneklem
# büyüklüğüyle karesel/kübik büyür. Bu yüzden SVR'nin hiper-parametre
# araması, eğitim setinden rastgele alınan 4.000 satırlık bir alt
# örneklem üzerinde yapılıyor; RF ve XGBoost tam eğitim setini
# (~13.278 satır) kullanıyor. Adil karşılaştırma için üç model de aynı,
# hiç dokunulmamış tam test setinde (~3.320 satır) değerlendiriliyor.

# %%
on_isleyici = ColumnTransformer([
    ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik_sutunlar),
    ("sayisal", StandardScaler(), sayisal_sutunlar),
])

cv_semasi = KFold(n_splits=5, shuffle=True, random_state=RASSAL_TOHUM)
skor = "neg_root_mean_squared_error"

modeller = {}
arama_sonuclari = {}

# --- 11.1 Random Forest ---
rf_pipeline = Pipeline([
    ("on_isleme", on_isleyici),
    ("model", RandomForestRegressor(random_state=RASSAL_TOHUM, n_jobs=2)),
])
rf_param_dagilimi = {
    "model__n_estimators": randint(150, 500),
    "model__max_depth": [None, 6, 10, 14, 20, 26],
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 10),
    "model__max_features": ["sqrt", "log2", 0.5, 0.7, None],
}
rf_arama = RandomizedSearchCV(
    rf_pipeline, rf_param_dagilimi, n_iter=18, cv=cv_semasi,
    scoring=skor, n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
rf_arama.fit(X_train, y_train_log)
modeller["Random Forest"] = rf_arama.best_estimator_
arama_sonuclari["Random Forest"] = rf_arama.best_params_
print("Random Forest en iyi parametreler:", rf_arama.best_params_)
print(f"CV RMSE (log ölçek): {-rf_arama.best_score_:.4f}")

# %%
# --- 11.2 SVR (Support Vector Regression) — 4.000 satırlık alt örneklem ---
X_train_svr = X_train.sample(n=4000, random_state=RASSAL_TOHUM)
y_train_svr_log = y_train_log.loc[X_train_svr.index]
print(f"SVR alt örneklem boyutu: {X_train_svr.shape[0]} satır "
      f"(tam eğitim setinin %{100*4000/len(X_train):.1f}'i)")

svr_pipeline = Pipeline([
    ("on_isleme", on_isleyici),
    ("model", SVR()),
])
svr_param_dagilimi = {
    "model__kernel": ["rbf"],
    "model__C": loguniform(1e-1, 1e2),
    "model__gamma": loguniform(1e-3, 1e0),
    "model__epsilon": uniform(0.01, 0.3),
}
svr_arama = RandomizedSearchCV(
    svr_pipeline, svr_param_dagilimi, n_iter=15, cv=cv_semasi,
    scoring=skor, n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
svr_arama.fit(X_train_svr, y_train_svr_log)
modeller["SVR"] = svr_arama.best_estimator_
arama_sonuclari["SVR"] = svr_arama.best_params_
print("SVR en iyi parametreler:", svr_arama.best_params_)
print(f"CV RMSE (log ölçek, 4.000 satırlık alt örneklem): {-svr_arama.best_score_:.4f}")

# %%
# --- 11.3 XGBoost Regressor ---
xgb_pipeline = Pipeline([
    ("on_isleme", on_isleyici),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=RASSAL_TOHUM, n_jobs=2)),
])
xgb_param_dagilimi = {
    "model__n_estimators": randint(150, 500),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__reg_alpha": loguniform(1e-3, 10),
    "model__reg_lambda": loguniform(1e-3, 10),
}
xgb_arama = RandomizedSearchCV(
    xgb_pipeline, xgb_param_dagilimi, n_iter=18, cv=cv_semasi,
    scoring=skor, n_jobs=2, random_state=RASSAL_TOHUM, verbose=0,
)
xgb_arama.fit(X_train, y_train_log)
modeller["XGBoost"] = xgb_arama.best_estimator_
arama_sonuclari["XGBoost"] = xgb_arama.best_params_
print("XGBoost en iyi parametreler:", xgb_arama.best_params_)
print(f"CV RMSE (log ölçek): {-xgb_arama.best_score_:.4f}")

# %% [markdown]
# ## 12. Test Setinde Değerlendirme ve Karşılaştırma Tablosu
#
# Her modelin tahmini `expm1` ile orijinal (milyon adet) ölçeğe geri
# dönüştürülüp negatif değerler 0'a kırpılıyor (satış negatif olamaz),
# sonra RMSE/MAE/R² bu ölçekte hesaplanıyor.

# %%
sonuc_satirlari = []
tahminler_ham = {}
for isim, model in modeller.items():
    tahmin_log = model.predict(X_test)
    tahmin_ham = np.clip(np.expm1(tahmin_log), 0, None)
    tahminler_ham[isim] = tahmin_ham
    rmse = np.sqrt(mean_squared_error(y_test_ham, tahmin_ham))
    mae = mean_absolute_error(y_test_ham, tahmin_ham)
    r2 = r2_score(y_test_ham, tahmin_ham)
    sonuc_satirlari.append({"Model": isim, "RMSE": rmse, "MAE": mae, "R2": r2})

karsilastirma_df = pd.DataFrame(sonuc_satirlari).sort_values("RMSE").reset_index(drop=True)
for c in ["RMSE", "MAE", "R2"]:
    karsilastirma_df[c] = karsilastirma_df[c].round(4)

karsilastirma_yolu = PROJE_KOK / "model_karsilastirma_tablosu.csv"
karsilastirma_df.to_csv(karsilastirma_yolu, index=False)
print(f"Kaydedildi: {karsilastirma_yolu.name}")
print(karsilastirma_df.to_string(index=False))

en_iyi_model_adi = karsilastirma_df.iloc[0]["Model"]
en_iyi_model = modeller[en_iyi_model_adi]
en_iyi_tahmin = tahminler_ham[en_iyi_model_adi]
print(f"\nRMSE'ye göre en iyi model: {en_iyi_model_adi}")

# %% [markdown]
# ## 13. Görsel 5 — Gerçek vs Tahmin (En İyi Model)
#
# Hedef 4 basamak (0,01 - 82,74 milyon) yayıldığı için eksenler log
# ölçekte çiziliyor; aksi halde neredeyse tüm noktalar sol-alt köşede
# üst üste binerdi ve grafik hiçbir şey anlatmazdı.

# %%
fig_tahmin = go.Figure()
fig_tahmin.add_trace(go.Scatter(
    x=y_test_ham, y=en_iyi_tahmin, mode="markers", name="Test Oyunları",
    marker=dict(color=RENK_MAVI, size=6, opacity=0.4),
    hovertemplate="Gerçek Satış: %{x:.2f} milyon adet<br>Tahmin Edilen Satış: %{y:.2f} milyon adet<extra></extra>",
))
sinir_min = max(float(min(y_test_ham.min(), en_iyi_tahmin.min())), 0.005)
sinir_max = float(max(y_test_ham.max(), en_iyi_tahmin.max()))
fig_tahmin.add_trace(go.Scatter(
    x=[sinir_min, sinir_max], y=[sinir_min, sinir_max], mode="lines",
    name="Mükemmel Tahmin (y=x)", line=dict(color=METIN_SOLUK, width=2, dash="dash"),
    hoverinfo="skip",  # referans çizgisi (y=x) — gösterilecek anlamlı veri yok
))
fig_tahmin.update_xaxes(type="log", title_text="Gerçek Küresel Satış (milyon adet, log ölçek)")
fig_tahmin.update_yaxes(type="log", title_text="Tahmin Edilen Satış (milyon adet, log ölçek)")
fig_tahmin.update_layout(PLOTLY_SABLON["layout"], title=f"Gerçek vs Tahmin — {en_iyi_model_adi}")
fig_tahmin.write_image(GORSEL_KOK / "05_gercek_vs_tahmin.png", width=950, height=700, scale=2)
fig_tahmin.write_html(GORSEL_KOK / "05_gercek_vs_tahmin.html")
print("Kaydedildi: 05_gercek_vs_tahmin.png / .html")

# %% [markdown]
# **Gözlem:** Noktalar y=x çizgisi etrafında toplansa da düşük satışlı
# oyunlarda dağılım daha geniş (mutlak hata küçük ama göreli hata büyük
# olabiliyor), yüksek satışlı birkaç blockbuster ise genelde hafif düşük
# tahmin ediliyor. Bu beklenen bir sonuç: Platform/Tür/Yayıncı/Yıl gibi
# "piyasaya çıkmadan bilinen" bilgiler bir oyunun kalitesini, pazarlama
# bütçesini veya viral etkisini yakalayamıyor — bunlar asıl blockbuster'ı
# blockbuster yapan şeyler ama veri setinde yok.

# %% [markdown]
# ## 14. Görsel 6 — Özellik Önemi

# %%
en_iyi_agac_adi = en_iyi_model_adi if en_iyi_model_adi in ("Random Forest", "XGBoost") else (
    karsilastirma_df[karsilastirma_df["Model"].isin(["Random Forest", "XGBoost"])].iloc[0]["Model"]
)
if en_iyi_agac_adi != en_iyi_model_adi:
    print(f"Not: En iyi model ({en_iyi_model_adi}) ağaç tabanlı değil, özellik önemi için "
          f"en iyi ağaç tabanlı model olan {en_iyi_agac_adi} kullanılıyor.")

onem_pipeline = modeller[en_iyi_agac_adi]
onem_model = onem_pipeline.named_steps["model"]
onem_ozellik_adlari = onem_pipeline.named_steps["on_isleme"].get_feature_names_out()
onem_degerleri = onem_model.feature_importances_


def grup_bul(ad: str) -> str:
    if "Publisher_Grup_" in ad:
        return "Yayıncı"
    if "Platform_" in ad:
        return "Platform"
    if "Genre_" in ad:
        return "Tür"
    return "Yıl"


onem_df = pd.DataFrame({
    "Özellik": [ad.split("__")[-1] for ad in onem_ozellik_adlari],
    "Grup": [grup_bul(ad) for ad in onem_ozellik_adlari],
    "Önem": onem_degerleri,
}).sort_values("Önem", ascending=False)

print("Gruba göre toplam özellik önemi:")
print(onem_df.groupby("Grup")["Önem"].sum().sort_values(ascending=False))

en_onemli_15 = onem_df.head(15).sort_values("Önem", ascending=True)
fig_onem = px.bar(
    en_onemli_15, x="Önem", y="Özellik", orientation="h", color="Grup",
    color_discrete_map={"Platform": RENK_MAVI, "Tür": RENK_TURUNCU, "Yayıncı": RENK_AQUA, "Yıl": RENK_MOR},
    title=f"Özellik Önemi — İlk 15 (en iyi ağaç tabanlı model: {en_iyi_agac_adi})",
    labels={"Önem": "Göreli Önem", "Özellik": ""},
)
fig_onem.update_layout(PLOTLY_SABLON["layout"])
fig_onem.update_traces(
    hovertemplate="Özellik: %{y}<br>Göreli Önem: %{x:.4f}<extra>%{fullData.name}</extra>"
)
fig_onem.write_image(GORSEL_KOK / "06_ozellik_onemi.png", width=1000, height=650, scale=2)
fig_onem.write_html(GORSEL_KOK / "06_ozellik_onemi.html")
print("Kaydedildi: 06_ozellik_onemi.png / .html")
print(en_onemli_15.sort_values("Önem", ascending=False).to_string(index=False))

# %% [markdown]
# **Gözlem:** Gruplara göre toplam önem şöyle sıralanıyor: Yayıncı ~%70,
# Platform ~%21, Tür ~%8, Yıl ~%1. Yani model kararını neredeyse tek
# başına yayıncı bilgisine dayandırıyor — bunun da büyük kısmı tek bir
# kategoriden geliyor: `Publisher_Grup_Nintendo` (0,33), listedeki ikinci
# en önemli özelliğin (Electronic Arts, 0,041) sekiz katından fazla. Bu
# Görsel 4'teki gözlemle tutarlı (Nintendo açık ara lider yayıncı) ama
# aynı zamanda bir uyarı: model büyük ölçüde "bu oyun Nintendo'nun mu
# değil mi" sorusuna indirgeniyor, `Yıl`'ın payı ise şaşırtıcı derecede
# düşük — çıkış yılı tek başına satışı neredeyse hiç açıklamıyor.

# %% [markdown]
# ## 15. Sonuç — Dürüst Değerlendirme
#
# Üç model de aynı veri, aynı tabakalı bölme, aynı çapraz doğrulama
# şeması (5 katlı, RandomizedSearchCV, log1p hedef) üzerinden ayarlandı.
# Karşılaştırma tablosu ve tüm görseller yukarıda üretildi.
#
# **Bunu süslemeden söylemek gerekiyor:** en iyi model olan XGBoost'un R²
# değeri sadece 0,097 — yani Platform, Tür, Yayıncı ve Yıl bilgisi,
# oyunların küresel satışındaki değişimin sadece ~%10'unu açıklayabiliyor.
# Bu bir modelleme hatası değil, veri setinin doğal bir sınırı: bir oyunun
# gerçekten ne kadar satacağını belirleyen asıl etkenler (oyunun kalitesi,
# pazarlama bütçesi, eleştirmen puanları, viral etki, franchise'ın o anki
# popülaritesi) bu veri setinde hiç yok. Sonuç olarak model "Nintendo'nun
# first-party bir oyunu ortalamadan daha çok satar" gibi kaba ama gerçek
# eğilimleri yakalıyor, ama "bu belirli oyun 40 milyon mu satar 4 milyon mu
# satar" sorusuna hassas cevap veremiyor. Ayrıntılı yorum README.md
# içinde.

# %%
print("Proje betiği tamamlandı.")
print("Üretilen görseller:", sorted(p.name for p in GORSEL_KOK.iterdir()))
