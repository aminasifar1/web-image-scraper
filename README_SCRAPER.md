# Advanced Web Image Scraper

## 📋 Descripción General

Un scraper de imágenes web **profesional y robusto** diseñado para:

✅ Extraer imágenes de **estructuras HTML complejas** (article, picture, source, video, etc.)  
✅ Excluir **completamente archivos .svg**  
✅ Eliminar **duplicados** (por URL y por hash)  
✅ Validar **calidad de imagen** con múltiples criterios  
✅ Generar **metadatos completos y detallados**  
✅ Exportar en **CSV y JSON** para fácil análisis  

## 🎯 Características Principales

### 1. Extracción Inteligente de Imágenes

El scraper busca imágenes en:

```html
<!-- Imágenes estándar -->
<img src="image.jpg" alt="Descripción">

<!-- Images responsive con srcset -->
<img src="small.jpg" srcset="medium.jpg 480w, large.jpg 1920w" alt="Responsive">

<!-- Picture tags (futuro de imágenes web) -->
<picture>
  <source media="(min-width: 768px)" srcset="desktop.jpg">
  <source srcset="mobile.jpg">
  <img src="fallback.jpg" alt="Picture">
</picture>

<!-- Lazy loading -->
<img data-src="lazy.jpg" loading="lazy" alt="Lazy Loaded">

<!-- Dentro de contenedores -->
<article>
  <img src="content.jpg" alt="Artículo">
</article>

<figure>
  <img src="figure.jpg" alt="Figura">
  <figcaption>Descripción</figcaption>
</figure>

<!-- Video posters -->
<video poster="poster.jpg">
  <source src="video.mp4">
</video>
```

### 2. Deduplicación Automática

- **Por URL**: No descarga la misma URL dos veces
- **Por contenido**: Detecta imágenes idénticas con diferente URL usando SHA256

### 3. Filtros de Calidad Automáticos

Rechaza automáticamente:

| Criterio | Rango Aceptable |
|----------|-----------------|
| Dimensión mínima | ≥ 100x100 píxeles |
| Área mínima | ≥ 10,000 píxeles totales |
| Dimensión máxima | ≤ 8000x8000 píxeles |
| Tamaño archivo | 5 KB - 10 MB |
| Formato | NO SVG (excluido) |

### 4. Metadatos Completos

Cada imagen incluye:

```json
{
  "image_id": "abc123def456",
  "image_url": "https://ejemplo.com/image.jpg",
  "source_domain": "ejemplo.com",
  "source_url": "https://ejemplo.com/articulo",
  "title": "Título",
  "alt_text": "Descripción",
  "image_type": "image/jpeg",
  "filename": "abc123def456.jpg",
  "width": 1920,
  "height": 1080,
  "file_size_kb": 245.3,
  "image_hash": "sha256hash...",
  "html_tag": "img",
  "parent_tag": "article",
  "classes": "article-image responsive",
  "element_id": "main-image",
  "image_timestamp": "2020-12-15T14:30:45",  // 📅 Fecha EXIF o Last-Modified
  "image_date": "2020-12-15",                // Solo la fecha
  "loading_strategy": "lazy"
}
```

## 🚀 Instalación Rápida

### 1. Clonar/Descargar

```bash
cd /tu/directorio/proyecto
```

### 2. Instalar Dependencias

```bash
pip install -r requirements_scraper.txt
```

O instalación manual:

```bash
pip install requests beautifulsoup4 pillow
```

### 3. Usar

```bash
# Uso básico
python advanced_image_scraper.py --url https://ejemplo.com

# Con opciones
python advanced_image_scraper.py \
    --url https://noticias.com \
    --output-dir ./mis_resultados \
    --max-pages 10 \
    --delay 2.0
```

## 📁 Estructura de Salida

```
image_scraper_results/
├── images/
│   ├── abc123def456.jpg          # Imágenes descargadas
│   ├── xyz789abc123.png
│   └── ...
│
└── metadata/
    ├── images_metadata.csv        # Todos los metadatos (CSV)
    ├── images_metadata.json       # Todos los metadatos (JSON)
    ├── images_ejemplo_com.csv    # Metadatos por dominio
    ├── summary.json               # Estadísticas generales
    └── ...
```

## 📊 Ejemplos de Uso

### Ejemplo 1: Scrapear Sitio de Noticias

```bash
python advanced_image_scraper.py \
    --url https://www.bbc.com/news \
    --max-pages 20 \
    --delay 2.0 \
    --output-dir ./noticias_images
```

Resultado: ~500-1000 imágenes con metadatos completos sobre origen, contexto, etc.

### Ejemplo 2: Analizar Metadatos

```python
import csv
import json

# Leer CSV
with open('metadata/images_metadata.csv', 'r') as f:
    images = list(csv.DictReader(f))

# Imágenes sin ALT (accesibilidad)
no_alt = [img for img in images if not img['alt_text']]
print(f"Imágenes sin ALT: {len(no_alt)}")

# Imágenes más grandes
large = sorted(images, key=lambda x: float(x['file_size_kb']), reverse=True)[:5]
for img in large:
    print(f"{img['filename']}: {img['file_size_kb']} KB")

# Dominios únicos
domains = set(img['source_domain'] for img in images)
print(f"Dominios: {', '.join(domains)}")
```

### Ejemplo 3: Filtrar por Criterios

```python
import json

with open('metadata/images_metadata.json') as f:
    images = json.load(f)

# Imágenes en alta resolución
hires = [img for img in images if img['width'] > 1920 and img['height'] > 1080]

# Imágenes responsive (picture tag)
responsive = [img for img in images if img['parent_tag'] == 'picture']

# Imágenes con lazy loading
lazy = [img for img in images if img['loading_strategy'] == 'lazy']

print(f"Alta resolución: {len(hires)}")
print(f"Responsive: {len(responsive)}")
print(f"Lazy loading: {len(lazy)}")
```

## 🔧 Opciones de Línea de Comandos

```
--url URL              URL del sitio a scrapear (REQUERIDO)
--output-dir PATH     Directorio de salida (default: ./image_scraper_results)
--max-pages N         Máximo de páginas a procesar (default: 1)
--delay SEGUNDOS      Delay entre peticiones (default: 1.0)
```

## 📈 Casos de Uso Reales

### 1. Investigación de Noticias
Descarga todas las imágenes de artículos con contexto para análisis.

### 2. Auditoría de Accesibilidad
Identifica imágenes sin ALT text o sin atributo 'title'.

### 3. Dataset para ML
Recopila imágenes con metadatos para entrenar modelos.

### 4. Análisis Competitivo
Examina imágenes usadas por competidores y sus características.

### 5. Investigación Visual
Analiza tendencias en uso de imágenes por tipo, resolución, formato.

## 📝 Notas Importantes

### ✅ Lo que SÍ hace

- ✓ Extrae imágenes de estructuras HTML complejas
- ✓ Valida calidad automáticamente
- ✓ Excluye .svg completamente
- ✓ Elimina duplicados
- ✓ Genera metadatos detallados
- ✓ Respetuoso con servidores (delay configurable)
- ✓ Robusto ante errores

### ❌ Lo que NO hace

- ✗ No ejecuta JavaScript (usa solo HTML)
- ✗ No descarga imágenes de CDN sin referencia en HTML
- ✗ No contorna robots.txt (si está prohibido, respeta)
- ✗ No maneja JavaScript renderizado dinámicamente

Para JavaScript dinámico, necesitarías Selenium/Playwright.

## 🎓 Ejemplos Incluidos

Archivo `examples.py` con 6 ejemplos educativos:

1. **Scraping Básico**: Descarga imágenes de una URL
2. **Análisis de Metadatos**: Lee y analiza CSV
3. **Exportar Filtrados**: Extrae subconjuntos específicos
4. **Análisis de Calidad**: Examina dimensiones y formato
5. **Reporte Accesibilidad**: Auditoría de ALT text
6. **Múltiples Páginas**: Ejemplo de scraping más completo

Ejecuta:
```bash
python examples.py
```

## 🐛 Troubleshooting

### Imágenes insuficientes

```bash
# Aumenta max-pages
python advanced_image_scraper.py --url https://ejemplo.com --max-pages 10

# Reduce delay para ser más rápido
python advanced_image_scraper.py --url https://ejemplo.com --delay 0.5
```

### Error: No module named 'requests'

```bash
pip install requests beautifulsoup4 pillow
```

### El sitio bloquea el bot

Algunos sitios detectan bots. Verifica:
1. El sitio permite scraping en robots.txt
2. Respeta el User-Agent (el scraper usa uno estándar)
3. Aumenta el delay (`--delay 3.0`)

## 📊 Análisis Post-Scraping con Pandas

```python
import pandas as pd

df = pd.read_csv('metadata/images_metadata.csv')

# Top 10 imágenes más grandes
print(df.nlargest(10, 'file_size_kb')[['filename', 'file_size_kb', 'alt_text']])

# Distribución de tipos
print(df['image_type'].value_counts())

# Imágenes sin ALT
print(f"Sin ALT: {(df['alt_text'].isna() | (df['alt_text'] == '')).sum()}")

# Correlación entre tamaño y dimensiones
print(df[['width', 'height', 'file_size_kb']].describe())
```

## 🤝 Contribuciones

¿Mejoras sugeridas?

1. Soporte para Selenium (JavaScript)
2. Inteligencia artificial para clasificar imágenes
3. Integración con bases de datos
4. API REST

## 📄 Licencia

Uso educativo y de investigación. Respeta siempre:
- robots.txt del sitio
- Términos de servicio
- Leyes locales de scraping

## 🎉 ¡Listo!

Tienes un scraper profesional de imágenes. ¡Usa responsablemente!

---

**Creado para:** Análisis web, investigación, datasets, auditoría  
**Última actualización:** 2024-04-22  
**Estado:** ✅ Funcional y listo para producción
