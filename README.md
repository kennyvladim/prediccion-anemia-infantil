# Predicción de Anemia Infantil con Machine Learning
### Pipeline v3 — ENDES 2024 Perú

Pipeline de aprendizaje automático para predecir anemia en niños menores de 5 años usando los microdatos de la Encuesta Demográfica y de Salud Familiar (ENDES 2024) del INEI Perú.

---

## Modelos implementados

| Modelo | Descripción |
|---|---|
| Random Forest | Ensamble de árboles con alta precisión |
| Regresión Logística | Modelo lineal interpretable |
| Árbol de Decisión | Modelo visual y explicable |

---

## Requisitos

- Python 3.9 o superior
- Los 9 archivos CSV de ENDES 2024 (ver sección de datos)

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

---

## Datos necesarios

Coloca los siguientes archivos CSV de ENDES 2024 en la carpeta `data/`:

```
data/
├── RECH0_2024.csv    — Características del hogar
├── RECH4_2024.csv    — Miembros del hogar
├── RECH5_2024.csv    — Hemoglobina materna
├── RECH6_2024.csv    — Niños menores de 5 años (módulo principal)
├── RECH23_2024.csv   — Riqueza y servicios
├── REC41_2024.csv    — Último nacimiento
├── REC43_2024.csv    — Salud del niño
├── REC94_2024.csv    — SIS del niño
└── REC95_2024.csv    — Suplementación Perú
```

> Los datos ENDES son de acceso público y pueden descargarse desde el portal del [INEI](https://proyectos.inei.gob.pe/microdatos/).

> ⚠️ Los archivos CSV **no se incluyen** en este repositorio por su tamaño.

---

## Uso

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/prediccion-anemia-infantil.git
cd prediccion-anemia-infantil
```

2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Coloca los 9 CSV de ENDES 2024 en la carpeta `data/`

4. Ejecuta el pipeline:
```bash
python Prediccion_Anemia_Infantil_ML.py
```

5. Los gráficos se guardan automáticamente en la carpeta `output/`

---

## Estructura del Pipeline

| Paso | Descripción |
|---|---|
| 01 | Carga de los 9 módulos CSV |
| 02 | Control de calidad y filtros |
| 03 | Merges entre módulos |
| 04 | Ingeniería de variables (29 predictores) |
| 05 | Entrenamiento de los 3 modelos |
| 06 | Evaluación y métricas |
| 07 | Inferencia en lote sobre 8 perfiles sintéticos |

---

## Predictores principales (29 variables)

- **Niño:** edad en meses, sexo, stunting, wasting
- **Geográfico:** altitud, área urbana/rural, región
- **Hogar:** índice de riqueza, agua mejorada, saneamiento, combustible limpio, número de niños
- **Madre:** educación, empleo, SIS, edad, hemoglobina
- **Prenatal:** control prenatal temprano, suplemento de hierro, peso al nacer
- **Morbilidad:** diarrea, tos, IRA
- **Intervenciones:** vitamina A, antiparasitario, SIS niño, hierro 7 días, micronutrientes MINSA, CRED

---

## Salidas generadas

- `output/grafico_1_*.png` — Distribución de la muestra
- `output/grafico_2_*.png` — Importancia de variables
- `output/grafico_3_*.png` — Métricas de evaluación
- `output/grafico_4_batch_inference.png` — Probabilidades por perfil sintético

---

## Fuente de datos

Instituto Nacional de Estadística e Informática (INEI) — Encuesta Demográfica y de Salud Familiar ENDES 2024.

---

## Licencia

MIT License — libre para uso académico y de investigación.
