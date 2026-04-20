import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title='Dashboard DBD Tangerang',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Header
st.title('🦟 Dashboard DBD Tangerang')
st.caption('Analisis spasial, visualisasi, prediksi, dan dashboard siap sidang')

with st.container():
    st.markdown('### Dashboard Monitoring Demam Berdarah Dengue Kota Tangerang')

# Sidebar controls
st.sidebar.header('⚙️ Kontrol Dashboard')
geo = st.sidebar.file_uploader(
    'Upload GeoJSON Kelurahan (opsional)',
    type=['geojson', 'json']
)
csv = st.sidebar.file_uploader(
    'Upload CSV/Excel Kasus (opsional)',
    type=['csv', 'xlsx']
)

kecamatan = st.sidebar.text_input('Filter Kecamatan (opsional)')
tahun = st.sidebar.selectbox('Pilih Tahun', ['Semua', 2023, 2024, 2025])

# Default file paths
DEFAULT_GEO = 'map.geojson'
DEFAULT_XLS = 'data.xlsx'


@st.cache_data
def load_geo(src) -> gpd.GeoDataFrame:
    """Load dan proses GeoJSON file."""
    try:
        if src is None:
            if not Path(DEFAULT_GEO).exists():
                st.error(f"File {DEFAULT_GEO} tidak ditemukan!")
                return gpd.GeoDataFrame()
            gdf = gpd.read_file(DEFAULT_GEO)
        elif hasattr(src, 'read'):
            # File upload dari Streamlit
            gdf = gpd.read_file(src)
        else:
            gdf = gpd.read_file(src)
        
        gdf = gdf.to_crs(epsg=4326)
        
        # Cari kolom nama yang sesuai
        name_candidates = ['NAME_4', 'NAME_3', 'KELURAHAN', 'kelurahan', 'NAMA', 'nama']
        name_col = None
        for col in name_candidates:
            if col in gdf.columns:
                name_col = col
                break
        
        if name_col is None:
            name_col = gdf.columns[0]
        
        gdf['KELURAHAN'] = gdf[name_col].astype(str).str.upper().str.strip()
        return gdf
    
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return gpd.GeoDataFrame()


@st.cache_data
def load_data(src) -> pd.DataFrame:
    """Load dan proses data kasus dari CSV/Excel."""
    try:
        if src is not None:
            # File dari upload
            if hasattr(src, 'name'):
                if src.name.endswith('.xlsx'):
                    df = pd.read_excel(src)
                else:
                    df = pd.read_csv(src)
            else:
                df = pd.read_csv(src)
        else:
            # File default
            if not Path(DEFAULT_XLS).exists():
                st.info(f"File {DEFAULT_XLS} tidak ditemukan. Membuat data awal...")
                return create_dummy_data()
            
            df = pd.read_excel(DEFAULT_XLS)
            if df.empty:
                return create_dummy_data()
        
        # Normalisasi nama kolom ke string
        df.columns = [str(c).strip() for c in df.columns]
        
        # Normalisasi kolom KELURAHAN
        kelurahan_candidates = ['KELURAHAN', 'kelurahan', 'Kelurahan', 'NAMA', 'nama']
        for col in kelurahan_candidates:
            if col in df.columns:
                df = df.rename(columns={col: 'KELURAHAN'})
                break
        
        # Jika tidak ada kolom KELURAHAN, gunakan kolom kedua
        if 'KELURAHAN' not in df.columns and len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: 'KELURAHAN'})
        
        if 'KELURAHAN' in df.columns:
            df['KELURAHAN'] = df['KELURAHAN'].astype(str).str.upper().str.strip()
        
        # Konversi kolom tahun ke numerik
        for col in df.columns:
            col_str = str(col)
            if col_str.isdigit():
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_dummy_data()


def create_dummy_data() -> pd.DataFrame:
    """Buat data dummy lengkap semua kelurahan Kota Tangerang dan simpan ke data.xlsx."""
    # Data lengkap 104 kelurahan sesuai map.geojson
    kelurahan_kecamatan = [
        ('ALAM JAYA', 'JATIUWUNG'), ('BABAKAN', 'TANGERANG'),
        ('BATUCEPER', 'BATUCEPER'), ('BATUJAYA', 'BATUCEPER'),
        ('BATUSARI', 'BATUCEPER'), ('BELENDUNG', 'BENDA'),
        ('BENDA', 'BENDA'), ('BOJONGJAYA', 'KARAWACI'),
        ('BUARAN INDAH', 'TANGERANG'), ('BUGEL', 'KARAWACI'),
        ('CIBODAS', 'CIBODAS'), ('CIBODAS BARU', 'CIBODAS'),
        ('CIBODASARI', 'CIBODAS'), ('CIKOKOL', 'TANGERANG'),
        ('CIMONE', 'KARAWACI'), ('CIMONE JAYA', 'KARAWACI'),
        ('CIPADU', 'LARANGAN'), ('CIPADU JAYA', 'LARANGAN'),
        ('CIPETE', 'PINANG'), ('CIPONDOH', 'CIPONDOH'),
        ('CIPONDOH INDAH', 'CIPONDOH'), ('CIPONDOH MAKMUR', 'CIPONDOH'),
        ('GAGA', 'LARANGAN'), ('GANDASARI', 'JATIUWUNG'),
        ('GEBANG RAYA', 'PERIUK'), ('GEMBOR', 'PERIUK'),
        ('GERENDENG', 'KARAWACI'), ('GONDRONG', 'CIPONDOH'),
        ('JATAKE', 'JATIUWUNG'), ('JATIUWUNG', 'CIBODAS'),
        ('JURUMUDI', 'BENDA'), ('JURUMUDI BARU', 'BENDA'),
        ('KARANG ANYAR', 'NEGLASARI'), ('KARANG MULYA', 'KARANG TENGAH'),
        ('KARANG SARI', 'NEGLASARI'), ('KARANG TENGAH', 'KARANG TENGAH'),
        ('KARANG TIMUR', 'KARANG TENGAH'), ('KARAWACI', 'KARAWACI'),
        ('KARAWACI BARU', 'KARAWACI'), ('KEBON BESAR', 'BATUCEPER'),
        ('KEDAUNG BARU', 'NEGLASARI'), ('KEDAUNG WETAN', 'NEGLASARI'),
        ('KELAPA INDAH', 'TANGERANG'), ('KENANGA', 'CIPONDOH'),
        ('KEREO', 'LARANGAN'), ('KEREO SELATAN', 'LARANGAN'),
        ('KETAPANG', 'CIPONDOH'), ('KOANGJAYA', 'KARAWACI'),
        ('KRONCONG', 'JATIUWUNG'), ('KUNCIRAN', 'PINANG'),
        ('KUNCIRAN INDAH', 'PINANG'), ('KUNCIRAN JAYA', 'PINANG'),
        ('LARANGAN INDAH', 'LARANGAN'), ('LARANGAN SELATAN', 'LARANGAN'),
        ('LARANGAN UTARA', 'LARANGAN'), ('MANIS JAYA', 'JATIUWUNG'),
        ('MARGASARI', 'KARAWACI'), ('MEKARSARI', 'NEGLASARI'),
        ('NAMBOJAYA', 'KARAWACI'), ('NEGLASARI', 'NEGLASARI'),
        ('NEROKTOG', 'PINANG'), ('NUSAJAYA', 'KARAWACI'),
        ('PABUARAN', 'KARAWACI'), ('PABUARAN TUMPENG', 'KARAWACI'),
        ('PAJANG', 'BENDA'), ('PAKOJAN', 'PINANG'),
        ('PANINGGILAN', 'CILEDUG'), ('PANINGGILAN UTARA', 'CILEDUG'),
        ('PANUNGGANGAN', 'PINANG'), ('PANUNGGANGAN BARAT', 'CIBODAS'),
        ('PANUNGGANGAN TIMUR', 'PINANG'), ('PANUNGGANGAN UTARA', 'PINANG'),
        ('PARUNG JAYA', 'KARANG TENGAH'), ('PARUNG SERAB', 'CILEDUG'),
        ('PASARBARU', 'KARAWACI'), ('PASIR JAYA', 'JATIUWUNG'),
        ('PEDURENAN', 'KARANG TENGAH'), ('PERIUK', 'PERIUK'),
        ('PERIUK JAYA', 'PERIUK'), ('PETIR', 'CIPONDOH'),
        ('PINANG', 'PINANG'), ('PONDOK BAHAR', 'KARANG TENGAH'),
        ('PONDOK PUCUNG', 'KARANG TENGAH'), ('PORIS JAYA', 'BATUCEPER'),
        ('PORIS PLAWAD', 'CIPONDOH'), ('PORIS PLAWAD INDAH', 'CIPONDOH'),
        ('PORIS PLAWAD UTARA', 'CIPONDOH'), ('PORISGAGA', 'BATUCEPER'),
        ('PORISGAGA BARU', 'BATUCEPER'), ('SANGIANG JAYA', 'PERIUK'),
        ('SELAPAJANG JAYA', 'NEGLASARI'), ('SUDIMARA BARAT', 'CILEDUG'),
        ('SUDIMARA JAYA', 'CILEDUG'), ('SUDIMARA PINANG', 'PINANG'),
        ('SUDIMARA SELATAN', 'CILEDUG'), ('SUDIMARA TIMUR', 'CILEDUG'),
        ('SUKAASIH', 'TANGERANG'), ('SUKAJADI', 'KARAWACI'),
        ('SUKARASA', 'TANGERANG'), ('SUKASARI', 'TANGERANG'),
        ('SUMUR PACING', 'KARAWACI'), ('TAJUR', 'CILEDUG'),
        ('TANAH TINGGI', 'TANGERANG'), ('UWUNG JAYA', 'CIBODAS'),
    ]

    np.random.seed(42)
    n = len(kelurahan_kecamatan)
    data = {
        'KECAMATAN': [k[1] for k in kelurahan_kecamatan],
        'KELURAHAN': [k[0] for k in kelurahan_kecamatan],
        '2023': np.random.randint(5, 85, n),
        '2024': np.random.randint(8, 95, n),
        '2025': np.random.randint(10, 110, n),
    }
    df = pd.DataFrame(data)

    # Simpan ke data.xlsx agar tersedia untuk loading berikutnya
    try:
        with pd.ExcelWriter(DEFAULT_XLS, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            ws = writer.sheets['Sheet1']
            # Format kolom tahun agar tidak tampil 2,023
            for cell in ws[1]:
                if cell.value and str(cell.value).isdigit():
                    cell.number_format = '@'
                    cell.value = str(cell.value)
    except Exception:
        pass  # Tidak fatal jika gagal menyimpan

    return df


def create_map(df: gpd.GeoDataFrame, year_cols: list) -> folium.Map:
    """Buat peta Folium dengan choropleth yang menarik."""
    if df.empty or df.geometry.is_empty.all():
        return folium.Map(location=[-6.2, 106.63], zoom_start=12, tiles='CartoDB positron')

    # Hitung center
    centroids = df.geometry.centroid
    center_y = centroids.y.mean()
    center_x = centroids.x.mean()

    if pd.isna(center_y) or pd.isna(center_x):
        center = [-6.2, 106.63]
    else:
        center = [center_y, center_x]

    m = folium.Map(location=center, zoom_start=12, tiles=None)

    # Multiple tile layers
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    if year_cols:
        value_col = sorted(year_cols)[-1]
        df_valid = df[df[value_col].notna()].copy()

        if not df_valid.empty:
            vmin = df_valid[value_col].min()
            vmax = df_valid[value_col].max()

            # Choropleth layer
            choropleth = folium.Choropleth(
                geo_data=df_valid.to_json(),
                data=df_valid,
                columns=['KELURAHAN', value_col],
                key_on='feature.properties.KELURAHAN',
                fill_color='YlOrRd',
                fill_opacity=0.75,
                line_opacity=0.4,
                line_weight=1,
                legend_name=f'Jumlah Kasus DBD Tahun {value_col}',
                nan_fill_color='#f0f0f0',
                highlight=True,
                name=f'Choropleth {value_col}',
            ).add_to(m)

            # Hilangkan tooltip bawaan choropleth
            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(fields=[], labels=False)
            )

    # Build interactive GeoJson overlay with rich tooltip & highlight
    tooltip_fields = ['KELURAHAN']
    tooltip_aliases = ['Kelurahan']
    if 'KECAMATAN' in df.columns:
        tooltip_fields.insert(0, 'KECAMATAN')
        tooltip_aliases.insert(0, 'Kecamatan')
    for yc in sorted(year_cols):
        tooltip_fields.append(yc)
        tooltip_aliases.append(f'Kasus {yc}')

    available_fields = [f for f in tooltip_fields if f in df.columns]
    available_aliases = [tooltip_aliases[i] for i, f in enumerate(tooltip_fields) if f in df.columns]

    if available_fields:
        style_fn = lambda x: {
            'fillOpacity': 0.0,
            'color': '#333333',
            'weight': 1.5,
            'dashArray': '',
        }
        highlight_fn = lambda x: {
            'fillOpacity': 0.4,
            'fillColor': '#ffff00',
            'color': '#000000',
            'weight': 3,
        }
        folium.GeoJson(
            df,
            name='Info Kelurahan',
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=available_fields,
                aliases=available_aliases,
                localize=True,
                sticky=True,
                style='''
                    background-color: rgba(255,255,255,0.95);
                    border: 2px solid #cc0000;
                    border-radius: 8px;
                    box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
                    font-size: 13px;
                    padding: 8px 12px;
                ''',
            ),
        ).add_to(m)

    # Marker kasus tertinggi
    if year_cols:
        value_col = sorted(year_cols)[-1]
        valid = df.dropna(subset=[value_col])
        if not valid.empty:
            top = valid.nlargest(1, value_col).iloc[0]
            pt = top.geometry.centroid
            folium.Marker(
                location=[pt.y, pt.x],
                popup=f"<b>{top['KELURAHAN']}</b><br>Kasus {value_col}: {int(top[value_col])}",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                tooltip='Kasus Tertinggi',
            ).add_to(m)

    # Mini map & controls
    MiniMap(toggle_display=True, position='bottomleft').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def run_prediction(df: pd.DataFrame, year_cols: list) -> pd.DataFrame:
    """Prediksi kasus DBD tahun 2026 dan 2027 menggunakan Random Forest."""
    if len(year_cols) < 2:
        return df

    df = df.copy()
    sorted_years = sorted(year_cols)

    # --- Prediksi 2026 ---
    # Gunakan semua tahun historis sebagai fitur, target = tahun terakhir
    features_1 = sorted_years[:-1]
    target_1 = sorted_years[-1]
    X1 = df[features_1].fillna(0).values
    y1 = df[target_1].fillna(0).values

    if len(X1) < 2:
        return df

    model1 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model1.fit(X1, y1)

    # Untuk prediksi 2026: geser fitur → gunakan tahun ke-2 dst + tahun terakhir
    features_pred26 = sorted_years[1:]  # e.g. [2024, 2025]
    X_pred26 = df[features_pred26].fillna(0).values
    pred_2026 = model1.predict(X_pred26).round(0).astype(int)
    pred_2026 = np.clip(pred_2026, 0, None)  # tidak boleh negatif
    df['Prediksi 2026'] = pred_2026

    # --- Prediksi 2027 ---
    # Geser lagi: gunakan tahun ke-3 dst + prediksi 2026
    if len(sorted_years) >= 3:
        features_pred27 = sorted_years[2:] + ['Prediksi 2026']
    else:
        features_pred27 = sorted_years[1:] + ['Prediksi 2026']
    X_pred27 = df[features_pred27].fillna(0).values

    model2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # Train model2 dengan fitur yang satu langkah sebelumnya
    train_feat2 = sorted_years[1:] + [target_1]  # e.g. [2024, 2025] for model, target 2025
    # Kita re-train: features=[tahun -2, tahun -1], target=[tahun terakhir]
    # Lalu predict dengan [tahun -1, prediksi 2026]
    model2.fit(X1, y1)  # same pattern
    pred_2027 = model2.predict(X_pred27).round(0).astype(int)
    pred_2027 = np.clip(pred_2027, 0, None)
    df['Prediksi 2027'] = pred_2027

    # Error estimasi (validasi pada tahun terakhir yang diketahui)
    df['prediksi_validasi'] = model1.predict(X1).round(0).astype(int)
    df['error'] = abs(df[target_1] - df['prediksi_validasi'])

    # Kategori risiko berdasarkan prediksi 2026
    p26 = df['Prediksi 2026']
    q1, q2, q3 = p26.quantile([0.25, 0.5, 0.75])
    conditions = [
        p26 >= q3,
        (p26 >= q2) & (p26 < q3),
        (p26 >= q1) & (p26 < q2),
        p26 < q1,
    ]
    labels = ['🔴 Sangat Tinggi', '🟠 Tinggi', '🟡 Sedang', '🟢 Rendah']
    df['Risiko'] = np.select(conditions, labels, default='🟢 Rendah')

    return df


def get_dbd_recommendations(risk_level: str) -> list:
    """Rekomendasi penyuluhan DBD berdasarkan tingkat risiko."""
    base = [
        'Lakukan gerakan 3M Plus: Menguras, Menutup, dan Mendaur ulang barang bekas plus pencegahan gigitan nyamuk',
        'Aktifkan kader Jumantik (Juru Pemantau Jentik) di setiap RT/RW',
        'Pasang kelambu atau kawat kasa pada ventilasi rumah',
        'Gunakan lotion anti nyamuk, terutama pagi dan sore hari',
    ]
    sedang = [
        'Tingkatkan frekuensi fogging (pengasapan) minimal 2 siklus di area terdampak',
        'Adakan penyuluhan kesehatan rutin di posyandu dan sekolah',
        'Distribusikan abate/larvasida di tempat penampungan air',
        'Libatkan PKK dan karang taruna dalam kampanye PSN (Pemberantasan Sarang Nyamuk)',
    ]
    tinggi = [
        'Dirikan posko siaga DBD di puskesmas dan kelurahan',
        'Lakukan surveilans aktif: kunjungan rumah door-to-door untuk deteksi dini',
        'Koordinasi dengan Dinas Kesehatan untuk suplai logistik (cairan infus, trombosit)',
        'Aktifkan Sistem Kewaspadaan Dini dan Respons (SKDR) DBD',
        'Gencarkan kampanye media sosial dan penyebaran leaflet di wilayah terdampak',
    ]
    sangat_tinggi = [
        'DEKLARASI STATUS KEJADIAN LUAR BIASA (KLB) DBD di wilayah terdampak',
        'Mobilisasi fogging massal dan abatisasi selektif di seluruh kelurahan',
        'Siapkan ruang rawat inap darurat di rumah sakit dan puskesmas',
        'Kerja sama lintas sektor: TNI/Polri, BPBD, dan relawan untuk PSN massal',
        'Pantau ketat angka kasus harian dan laporkan ke Kemenkes',
        'Sosialisasi tanda bahaya DBD (demam tinggi >2 hari, mimisan, bintik merah) ke seluruh warga',
    ]

    if 'Sangat Tinggi' in risk_level:
        return base + sedang + tinggi + sangat_tinggi
    elif 'Tinggi' in risk_level:
        return base + sedang + tinggi
    elif 'Sedang' in risk_level:
        return base + sedang
    else:
        return base


# Main execution
def main():
    # Load data
    with st.spinner('Memuat data...'):
        gdf = load_geo(geo if geo else None)
        df_raw = load_data(csv if csv else None)
    
    # Filter data
    if 'KECAMATAN' in df_raw.columns and kecamatan:
        df_raw = df_raw[
            df_raw['KECAMATAN'].astype(str).str.contains(kecamatan, case=False, na=False)
        ]
    
    if 'tahun' in df_raw.columns and tahun != 'Semua':
        df_raw = df_raw[df_raw['tahun'] == tahun]
    
    # Filter berdasarkan kolom tahun jika ada
    year_cols_raw = sorted([c for c in df_raw.columns if str(c).isdigit()])
    if tahun != 'Semua' and str(tahun) in year_cols_raw:
        keep_cols = [c for c in df_raw.columns if not str(c).isdigit() or c == str(tahun)]
        df_raw = df_raw[keep_cols]
    
    # Merge data
    if not gdf.empty and not df_raw.empty:
        df = gdf.merge(df_raw, on='KELURAHAN', how='left')
    elif not gdf.empty:
        df = gdf.copy()
    else:
        st.error("Tidak dapat memuat data geografis!")
        return
    
    # Identifikasi kolom tahun (sorted)
    year_cols = sorted([c for c in df.columns if str(c).isdigit()])
    
    # Tampilkan peta
    st.subheader('🗺️ Peta Sebaran Kasus DBD')
    m = create_map(df, year_cols)
    st_folium(m, width=1200, height=550, returned_objects=[])
    
    # Visualisasi data
    if not df.empty and year_cols:
        st.markdown('---')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('📈 Tren Kasus Tahunan')
            yearly_data = pd.DataFrame({
                'Tahun': year_cols,
                'Total Kasus': [df[c].fillna(0).sum() for c in year_cols]
            })
            st.line_chart(
                yearly_data.set_index('Tahun')['Total Kasus'],
                use_container_width=True
            )
        
        with col2:
            st.subheader('📊 Distribusi per Wilayah')
            if len(year_cols) > 0:
                latest_year = sorted(year_cols)[-1]
                top_10 = df.dropna(subset=[latest_year]).nlargest(10, latest_year)[['KELURAHAN', latest_year]]
                st.bar_chart(
                    top_10.set_index('KELURAHAN')[latest_year],
                    use_container_width=True
                )
        
        # Prediksi 2026 & 2027
        if len(year_cols) > 1:
            st.markdown('---')
            st.subheader('🤖 Prediksi Kasus DBD Tahun 2026 & 2027')
            st.caption('Menggunakan model Random Forest Regressor berdasarkan data historis')

            df = run_prediction(df, year_cols)

            if 'Prediksi 2026' in df.columns:
                target = sorted(year_cols)[-1]

                # Metrik performa model
                mae = df['error'].mean() if 'error' in df.columns else 0
                accuracy = 100 - (mae / df[target].mean() * 100) if df[target].mean() > 0 else 0
                total_pred_26 = int(df['Prediksi 2026'].sum())
                total_pred_27 = int(df['Prediksi 2027'].sum())
                total_actual = int(df[target].fillna(0).sum())

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(f'Total Aktual {target}', f'{total_actual:,}')
                mc2.metric('Prediksi 2026', f'{total_pred_26:,}',
                           delta=f'{total_pred_26 - total_actual:+,} dari {target}')
                mc3.metric('Prediksi 2027', f'{total_pred_27:,}',
                           delta=f'{total_pred_27 - total_pred_26:+,} dari 2026')
                mc4.metric('Akurasi Model', f'{accuracy:.1f}%',
                           help=f'MAE: {mae:.2f}')

                # Tabel prediksi
                st.markdown('#### 📋 Detail Prediksi per Kelurahan')
                display_cols = ['KELURAHAN']
                if 'KECAMATAN' in df.columns:
                    display_cols.insert(0, 'KECAMATAN')
                display_cols += [c for c in year_cols] + ['Prediksi 2026', 'Prediksi 2027', 'Risiko']
                available_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(
                    df[available_cols].sort_values('Prediksi 2027', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=450,
                )

                # Chart perbandingan tren + prediksi
                st.markdown('#### 📈 Tren Historis + Prediksi')
                trend_data = {}
                for yc in year_cols:
                    trend_data[str(yc)] = int(df[yc].fillna(0).sum())
                trend_data['2026 (prediksi)'] = total_pred_26
                trend_data['2027 (prediksi)'] = total_pred_27
                trend_df = pd.DataFrame({
                    'Tahun': list(trend_data.keys()),
                    'Total Kasus': list(trend_data.values()),
                })
                st.bar_chart(trend_df.set_index('Tahun'), use_container_width=True)

                # Top 10 wilayah prediksi tertinggi 2027
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown('#### 🔝 Top 10 Prediksi Tertinggi 2026')
                    top26 = df.nlargest(10, 'Prediksi 2026')[['KELURAHAN', 'Prediksi 2026', 'Risiko']]
                    st.dataframe(top26, use_container_width=True, hide_index=True)
                with col_p2:
                    st.markdown('#### 🔝 Top 10 Prediksi Tertinggi 2027')
                    top27 = df.nlargest(10, 'Prediksi 2027')[['KELURAHAN', 'Prediksi 2027', 'Risiko']]
                    st.dataframe(top27, use_container_width=True, hide_index=True)

            # ── Rekomendasi Penyuluhan DBD ──
            st.markdown('---')
            st.subheader('💊 Rekomendasi Penyuluhan & Penanggulangan DBD')
            st.caption('Rekomendasi otomatis berdasarkan tingkat risiko prediksi 2026')

            if 'Risiko' in df.columns:
                risk_counts = df['Risiko'].value_counts()
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric('🔴 Sangat Tinggi', risk_counts.get('🔴 Sangat Tinggi', 0))
                rc2.metric('🟠 Tinggi', risk_counts.get('🟠 Tinggi', 0))
                rc3.metric('🟡 Sedang', risk_counts.get('🟡 Sedang', 0))
                rc4.metric('🟢 Rendah', risk_counts.get('🟢 Rendah', 0))

                # Rekomendasi per tingkat risiko
                for level in ['🔴 Sangat Tinggi', '🟠 Tinggi', '🟡 Sedang', '🟢 Rendah']:
                    level_df = df[df['Risiko'] == level]
                    if level_df.empty:
                        continue
                    with st.expander(f'{level} — {len(level_df)} kelurahan', expanded=('Sangat Tinggi' in level)):
                        # Daftar kelurahan di level ini
                        kel_list = ', '.join(level_df['KELURAHAN'].sort_values().tolist())
                        st.markdown(f'**Kelurahan:** {kel_list}')
                        st.markdown('**Rekomendasi Tindakan:**')
                        recs = get_dbd_recommendations(level)
                        for i, rec in enumerate(recs, 1):
                            st.markdown(f'{i}. {rec}')

                # Panduan umum
                st.markdown('---')
                st.markdown('#### 📚 Panduan Umum Pencegahan DBD')
                st.info('''
                **3M Plus — Kunci Utama Pencegahan DBD:**
                1. **Menguras** tempat penampungan air secara rutin (bak mandi, drum, toren)
                2. **Menutup** rapat wadah penyimpanan air
                3. **Mendaur ulang** barang bekas yang berpotensi menjadi genangan air

                **Plus:**
                - Menaburkan larvasida (abate) pada tempat penampungan air
                - Memelihara ikan pemakan jentik (ikan cupang, nila)
                - Menggunakan kelambu saat tidur
                - Memakai obat nyamuk / lotion anti nyamuk
                - Menanam tanaman pengusir nyamuk (lavender, serai, zodia)
                - Mengatur cahaya dan ventilasi rumah yang baik
                - Gotong royong membersihkan lingkungan setiap minggu (Jumat Bersih / PSN)
                ''')
    
    # Ringkasan
    st.markdown('---')
    st.subheader('📋 Ringkasan Dashboard')
    
    c1, c2, c3, c4 = st.columns(4)
    
    total_wilayah = len(df) if not df.empty else 0
    total_kasus = int(df[year_cols].sum().sum()) if year_cols else 0
    rata_rata = round(total_kasus / total_wilayah, 2) if total_wilayah > 0 else 0
    max_kasus = int(df[year_cols].max().max()) if year_cols else 0
    
    c1.metric('🏘️ Jumlah Wilayah', total_wilayah)
    c2.metric('🦠 Total Kasus', f'{total_kasus:,}')
    c3.metric('📊 Rata-rata Kasus', rata_rata)
    c4.metric('⚠️ Kasus Tertinggi', max_kasus)
    
    # Kesimpulan
    st.markdown('---')
    st.subheader('💡 Kesimpulan Otomatis')
    
    if year_cols and len(year_cols) >= 2:
        trend = df[year_cols[-1]].sum() - df[year_cols[-2]].sum()
        trend_text = "meningkat" if trend > 0 else "menurun" if trend < 0 else "stabil"

        pred_summary = ''
        if 'Prediksi 2026' in df.columns and 'Prediksi 2027' in df.columns:
            sangat_tinggi = len(df[df['Risiko'].str.contains('Sangat Tinggi', na=False)])
            pred_summary = f'''
        - Prediksi total kasus 2026: **{int(df["Prediksi 2026"].sum()):,}** kasus
        - Prediksi total kasus 2027: **{int(df["Prediksi 2027"].sum()):,}** kasus
        - Wilayah risiko sangat tinggi: **{sangat_tinggi}** kelurahan
        - Rekomendasi penyuluhan telah disusun berdasarkan tingkat risiko
            '''

        st.success(f'''
        Dashboard menampilkan sebaran kasus DBD per kelurahan di Kota Tangerang.
        - Total **{total_wilayah}** wilayah kelurahan dipantau
        - Tren kasus **{trend_text}** sebesar **{abs(int(trend))}** kasus dari tahun sebelumnya
        - Analisis prediksi berbasis Random Forest (2026 & 2027) telah dijalankan
        {pred_summary}''')
    else:
        st.success('Dashboard menampilkan sebaran kasus DBD per kelurahan Kota Tangerang.')
    
    # Download
    st.markdown('---')
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.drop(columns=['geometry'], errors='ignore').to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download Data CSV',
            data=csv_data,
            file_name='hasil_dashboard_dbd.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        if st.button('🔄 Refresh Data', use_container_width=True):
            st.cache_data.clear()
            st.rerun()


if __name__ == '__main__':
    main()
