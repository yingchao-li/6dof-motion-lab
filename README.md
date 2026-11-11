# 6-DoF Motion Lab

[English](#english) · [Deutsch](#deutsch) · [中文](#中文)

## English

A lightweight 6-DoF data analysis and scientific visualization tool built with Python, Streamlit, and Plotly. It supports trajectory exploration, kinematics, spatial entropy, descriptive statistics, DBSCAN, KMeans, and PCA.

The interface is available in English, German, and Chinese. Use the language selector at the top of the sidebar to switch instantly. English is the default.

### Real datasets included in the project

The app provides four real-data demo options and limits the number of points loaded per trajectory for responsive interaction.

| Directory | Files and size | Actual columns | Processing |
|---|---|---|---|
| `dataset/` | `sequence1.csv` … `sequence4.csv`, approximately 6,100–7,000 points per file | `Frame, PosX, PosY, PosZ, RotX, RotY, RotZ, RotW` | Each file contains about 35 runs where `Frame` resets. They are split into independent trials automatically; the filename is used as the experimental condition. Default frame rate: 60 Hz. |
| `raw/` | `User1.csv` … `User5.csv`, approximately 1,500–4,900 points per file | `timestamp, x, y, z, qx, qy, qz, qw` | Timestamps are detected as .NET ticks at 100 ns per tick; the filename is used as participant ID. |

The project labels these sources as “ViewGauss” and “HoloLens” based on the supplied directories and schemas. Because the CSV files contain no coordinate-system metadata, the app preserves native XYZ by default. The sidebar provides explicit presets for:

- Native XYZ
- Unity left-handed to right-handed coordinates by flipping Z
- Y-up to Z-up coordinates using `X, Z, -Y`

Quaternions are read as `(qx, qy, qz, qw)`, validated, and normalized. Coordinate presets transform positions and orientations into the same basis. Angular speed uses the shortest arc between consecutive quaternions and correctly handles the equivalent `q` and `-q` representations.

### Features

- CSV upload with automatic column detection and manual mapping for time, position, quaternion, participant, and condition columns
- English, German, and Chinese interface, with English as the default
- Interactive 3D trajectories, start and end points, distance, speed, acceleration, and angular speed
- 3D-grid Shannon entropy, spatial coverage, and 2D occupancy heatmaps
- Descriptive statistics and distributions; non-parametric tests run only when each condition contains at least two independent participants or sequences
- DBSCAN clustering of spatial samples from the current trajectory
- KMeans and PCA for participant- or sequence-level behavioral features
- CSV downloads for processed samples, participant features, and clustering results
- Independent plot and DBSCAN point limits for smooth operation on ordinary laptops

### Installation and startup

Python 3.10–3.13 is recommended.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser normally opens `http://localhost:8501` automatically.

### Minimum CSV requirements

The CSV must contain columns that can be mapped to these eight required fields:

```text
time, x, y, z, qx, qy, qz, qw
```

Supported time formats are seconds, milliseconds, microseconds, nanoseconds, .NET ticks, and frame numbers. `participant` and `condition` are optional. If a file contains only one trajectory, the app treats it as one independent sample and skips inappropriate KMeans/PCA or inferential-statistics operations.

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover both real dataset schemas, automatic column mapping, .NET ticks, frame-rate conversion, quaternion sign equivalence, coordinate transforms, spatial entropy, independent-sample safeguards, and translation completeness.

---

## Deutsch

Ein leichtgewichtiges Werkzeug zur Analyse und wissenschaftlichen Visualisierung von 6-DoF-Daten, entwickelt mit Python, Streamlit und Plotly. Es unterstützt Trajektorienanalyse, Kinematik, räumliche Entropie, deskriptive Statistik, DBSCAN, KMeans und PCA.

Die Benutzeroberfläche ist auf Englisch, Deutsch und Chinesisch verfügbar. Die Sprache lässt sich oben in der Seitenleiste sofort umschalten. Standardsprache ist Englisch.

### Im Projekt enthaltene Echtdaten

Die Anwendung bietet vier Demooptionen mit realen Daten und begrenzt die Anzahl der geladenen Punkte pro Trajektorie, damit die Bedienung flüssig bleibt.

| Verzeichnis | Dateien und Umfang | Tatsächliche Spalten | Verarbeitung |
|---|---|---|---|
| `dataset/` | `sequence1.csv` … `sequence4.csv`, ungefähr 6.100–7.000 Punkte pro Datei | `Frame, PosX, PosY, PosZ, RotX, RotY, RotZ, RotW` | Jede Datei enthält etwa 35 Durchläufe, in denen `Frame` zurückgesetzt wird. Diese werden automatisch in unabhängige Trials aufgeteilt; der Dateiname dient als Versuchsbedingung. Standardbildrate: 60 Hz. |
| `raw/` | `User1.csv` … `User5.csv`, ungefähr 1.500–4.900 Punkte pro Datei | `timestamp, x, y, z, qx, qy, qz, qw` | Zeitstempel werden als .NET-Ticks mit 100 ns pro Tick erkannt; der Dateiname dient als Teilnehmer-ID. |

Das Projekt bezeichnet diese Quellen anhand der bereitgestellten Verzeichnisse und Schemata als „ViewGauss“ und „HoloLens“. Da die CSV-Dateien keine Metadaten zum Koordinatensystem enthalten, bleiben die nativen XYZ-Koordinaten standardmäßig unverändert. In der Seitenleiste stehen folgende Vorgaben zur Verfügung:

- Native XYZ-Koordinaten
- Unity linkshändig zu rechtshändig durch Spiegeln der Z-Achse
- Y-up zu Z-up mit `X, Z, -Y`

Quaternionen werden als `(qx, qy, qz, qw)` gelesen, geprüft und normalisiert. Koordinatenvorgaben transformieren Position und Orientierung in dieselbe Basis. Die Winkelgeschwindigkeit verwendet den kürzesten Bogen zwischen aufeinanderfolgenden Quaternionen und verarbeitet die äquivalenten Darstellungen `q` und `-q` korrekt.

### Funktionen

- CSV-Upload mit automatischer Spaltenerkennung und manueller Zuordnung für Zeit, Position, Quaternion, Teilnehmende und Versuchsbedingung
- Benutzeroberfläche auf Englisch, Deutsch und Chinesisch; Standardsprache ist Englisch
- Interaktive 3D-Trajektorien, Start- und Endpunkte, Strecke, Geschwindigkeit, Beschleunigung und Winkelgeschwindigkeit
- Shannon-Entropie in einem 3D-Gitter, räumliche Abdeckung und zweidimensionale Belegungs-Heatmaps
- Deskriptive Statistik und Verteilungen; nichtparametrische Tests werden nur ausgeführt, wenn jede Bedingung mindestens zwei unabhängige Teilnehmende oder Sequenzen enthält
- DBSCAN-Clustering der Raumpunkte der aktuellen Trajektorie
- KMeans und PCA für Verhaltensmerkmale auf Teilnehmer- oder Sequenzebene
- CSV-Downloads für verarbeitete Messpunkte, Teilnehmermerkmale und Clusterergebnisse
- Separate Punktgrenzen für Diagramme und DBSCAN für eine flüssige Ausführung auf gewöhnlichen Laptops

### Installation und Start

Empfohlen wird Python 3.10–3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Der Browser öffnet normalerweise automatisch `http://localhost:8501`.

### Mindestanforderungen an die CSV-Datei

Die CSV-Datei muss Spalten enthalten, die den folgenden acht Pflichtfeldern zugeordnet werden können:

```text
time, x, y, z, qx, qy, qz, qw
```

Unterstützt werden Sekunden, Millisekunden, Mikrosekunden, Nanosekunden, .NET-Ticks und Frame-Nummern. `participant` und `condition` sind optional. Enthält eine Datei nur eine Trajektorie, behandelt die Anwendung sie als eine unabhängige Stichprobe und überspringt ungeeignete KMeans-/PCA- oder Inferenzstatistik-Schritte.

### Tests

```bash
python -m unittest discover -s tests -v
```

Die Tests decken beide realen Datenschemata, die automatische Spaltenzuordnung, .NET-Ticks, Bildratenumrechnung, Quaternion-Vorzeichenäquivalenz, Koordinatentransformationen, räumliche Entropie, Schutzregeln für unabhängige Stichproben und die Vollständigkeit der Übersetzungen ab.

---

## 中文

一个轻量级的 6-DoF 数据分析与科学可视化工具，使用 Python、Streamlit 与 Plotly 构建，支持轨迹分析、运动学、空间熵、描述统计、DBSCAN、KMeans 与 PCA。

界面支持英语、德语和中文，可在侧栏顶部即时切换；首次打开默认显示英语。

### 项目内真实数据

应用提供四个真实数据演示入口，并限制每条轨迹的读取点数，保证交互流畅。

| 目录 | 实际文件与规模 | 实际字段 | 应用处理 |
|---|---|---|---|
| `dataset/` | `sequence1.csv` … `sequence4.csv`，每个文件约 6,100–7,000 点 | `Frame, PosX, PosY, PosZ, RotX, RotY, RotZ, RotW` | 每个文件含约 35 个 `Frame` 重置段；自动拆成独立 trial，文件名作为实验条件；默认 60 Hz。 |
| `raw/` | `User1.csv` … `User5.csv`，每个文件约 1,500–4,900 点 | `timestamp, x, y, z, qx, qy, qz, qw` | 自动识别为 .NET ticks，每 tick 为 100 ns；文件名作为参与者 ID。 |

项目根据提供的目录和字段结构将两套数据标记为 “ViewGauss” 和 “HoloLens”。由于源 CSV 没有附带坐标系元数据，应用默认保留原始 XYZ，避免擅自改变数据。侧栏提供以下显式坐标预设：

- 保持原始 XYZ
- Unity 左手系到右手系，翻转 Z 轴
- Y-up 到 Z-up，使用 `X, Z, -Y`

四元数统一按 `(qx, qy, qz, qw)` 读取、检查并归一化。选择坐标预设时，位置和姿态都会转换到同一基底。角速度使用相邻四元数的最短弧，可正确处理等价的 `q` 与 `-q` 表示。

### 功能

- 上传 CSV，自动推断常见列名，也可手动映射时间、位置、四元数、参与者和实验条件
- 英语、德语、中文三语界面，默认英语
- 3D 交互轨迹、起终点、距离、速度、加速度与角速度
- 3D 网格 Shannon Entropy、空间覆盖率和二维占用热力图
- 描述统计与分布；只有每个条件至少包含 2 个独立参与者或序列时，才进行非参数检验
- DBSCAN 对当前轨迹的空间采样点聚类
- KMeans 与 PCA 对参与者或序列级行为特征建模
- 下载处理后的采样点、参与者特征和聚类结果 CSV
- 绘图与 DBSCAN 独立限采样，保证普通笔记本流畅运行

### 安装与启动

建议使用 Python 3.10–3.13。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。

### 上传 CSV 最低要求

CSV 至少要能映射到以下 8 个必需字段：

```text
time, x, y, z, qx, qy, qz, qw
```

时间支持秒、毫秒、微秒、纳秒、.NET ticks 和帧号。`participant` 与 `condition` 可选。一个文件只有一条轨迹时，应用会将其视为单个独立样本，并跳过不适用的 KMeans/PCA 或推断统计操作。

### 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖两套真实数据字段、自动列映射、.NET ticks、帧率换算、四元数符号等价、坐标变换、空间熵、独立样本统计保护以及翻译完整性。
