# Advanced Image Scraper - Guía de Uso

## Descripción General

El **Advanced Image Scraper** es un scraper de imágenes web avanzado que:

✅ **Extrae imágenes de múltiples estructuras HTML:**
- `<img>` tags estándar
- `<picture>` con `<source>` y `srcset` (responsive images)
- `<article>` y otros contenedores con imágenes
- `<video>` con posters
- Data attributes (lazy loading)

✅ **Excluye .svg** automáticamente

✅ **Elimina duplicados:**
- Por URL
- Por hash SHA256 (mismo contenido)

✅ **Filtros de calidad:**
- Dimensiones mínimas (100x100 px)
- Área mínima (10,000 píxeles)
- Tamaño archivo (5 KB - 10 MB)
- Validación de formato

✅ **Metadatos detallados:**
- Label/Dominio de origen
- Título y ALT text
- Tipo MIME de imagen
- Fecha y hora de descarga
- Dimensiones reales
- Hash para deduplicación
- Información de estructura HTML
- Estrategia de carga (lazy/eager)

## Instalación

### 1. Instalar dependencias requeridas

```bash
pip install requests beautifulsoup4 pillow
```

### 2. (Opcional) Crear un requirements.txt

```bash
echo "requests>=2.28.0
beautifulsoup4>=4.11.0
pillow>=9.2.0" > requirements_scraper.txt

pip install -r requirements_scraper.txt
```

## Uso Básico

### Línea de comandos simple

```bash
python advanced_image_scraper.py --url https://ejemplo.com
```

### Con más opciones

```bash
python advanced_image_scraper.py \
    --url https://noticias.ejemplo.com \
    --output-dir ./resultados_noticias \
    --max-pages 5 \
    --delay 2.0
```

### Opciones disponibles

| Opción | Descripción | Default |
|--------|------------|---------|
| `--url` | URL del sitio a scrapear | **Requerido** |
| `--output-dir` | Directorio donde guardar resultados | `./image_scraper_results` |
| `--max-pages` | Número máximo de páginas a procesar | `1` |
| `--delay` | Delay entre peticiones (segundos) | `1.0` |

## Estructura de Salida

```
image_scraper_results/
├── images/                          # Imágenes descargadas
│   ├── abc123def456.jpg
│   ├── xyz789abc123.png
│   └── ...
│
└── metadata/                        # Metadatos en varios formatos
    ├── images_metadata.csv          # CSV principal con todos los metadatos
    ├── images_metadata.json         # JSON detallado
    ├── images_example.com_img.csv  # CSV por dominio
    ├── summary.json                 # Reporte resumen
    └── ...
```

## Formato de Metadatos

Cada imagen descargada incluye los siguientes metadatos en CSV/JSON:

```
{
  "image_id": "abc123def456789",
  "image_url": "https://ejemplo.com/images/foto.jpg",
  "source_domain": "ejemplo.com",
  "source_url": "https://ejemplo.com/articulo",
  "title": "Título de la imagen",
  "alt_text": "Descripción alternativa",
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
  "image_timestamp": "2020-12-15T14:30:45",
  "image_date": "2020-12-15",
  "image_name_from_url": "foto.jpg",
  "srcset": "foto-small.jpg 480w, foto-large.jpg 1920w",
  "loading_strategy": "lazy"
}
```

## Ejemplos de Uso

### Ejemplo 1: Scrapear un sitio de noticias

```bash
python advanced_image_scraper.py \
    --url https://www.ejemplo-noticias.com \
    --output-dir ./noticias_images \
    --max-pages 10 \
    --delay 2.0
```

Este comando:
- Descargará imágenes de hasta 10 páginas
- Respetará un delay de 2 segundos entre peticiones
- Guardará todo en `./noticias_images`
- Generará metadatos con dominio, títulos, ALTs, etc.

### Ejemplo 2: Scrapear página única de blog

```bash
python advanced_image_scraper.py --url https://blog.ejemplo.com/articulo
```

### Ejemplo 3: Análisis de metadatos después del scraping

```python
import csv
import json

# Leer CSV de metadatos
with open('./image_scraper_results/metadata/images_metadata.csv', 'r') as f:
    reader = csv.DictReader(f)
    images = list(reader)
    
    # Filtrar imágenes por tamaño
    large_images = [img for img in images if int(img['width']) > 1000]
    
    # Agrupar por dominio
    by_domain = {}
    for img in images:
        domain = img['source_domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(img)

# Leer JSON para análisis
with open('./image_scraper_results/metadata/images_metadata.json', 'r') as f:
    all_images = json.load(f)
    
    # Imágenes sin ALT text (accesibilidad)
    missing_alt = [img for img in all_images if not img['alt_text']]
    print(f"Imágenes sin ALT: {len(missing_alt)}")
```

## Filtros de Calidad

El scraper rechaza automáticamente imágenes que:

- **Sean .svg** (como solicitaste)
- **Tengan dimensiones muy pequeñas**: menos de 100x100 píxeles
- **Tengan área pequeña**: menos de 10,000 píxeles totales
- **Sean muy grandes**: más de 8000x8000 píxeles o 10 MB
- **Tengan tamaño muy pequeño**: menos de 5 KB
- **No sean imágenes válidas**: formato inválido o corrupto

## Fecha de la Imagen

El scraper extrae la fecha de la imagen de tres fuentes posibles (en orden de prioridad):

1. **Metadatos EXIF**: Busca en DateTime, DateTimeOriginal, DateTimeDigitized
2. **Header Last-Modified**: Obtiene la fecha del servidor HTTP
3. **Desconocida**: Si no encuentra fecha en ningún lugar, deja el campo vacío

Esta información se almacena en dos campos:
- `image_timestamp`: Timestamp completo ISO (ej: `2020-12-15T14:30:45`)
- `image_date`: Solo la fecha en formato YYYY-MM-DD (ej: `2020-12-15`)

El resumen final también incluye un rango de fechas (`date_range`) con la imagen más antigua y más nueva encontrada.

## Deduplicación

El scraper elimina duplicados de dos formas:

1. **Por URL**: No descarga la misma URL dos veces
2. **Por contenido**: Si dos URLs diferentes tienen el mismo contenido (mismo hash SHA256), solo guarda una

Esto es útil porque evita descargar la misma imagen alojada en múltiples CDNs o con URLs diferentes.

## Estructura HTML Compleja

### Ejemplo 1: Picture con responsive images

```html
<picture>
  <source media="(min-width: 768px)" srcset="grande.jpg, grande-2x.jpg 2x">
  <source srcset="pequeña.jpg, pequeña-2x.jpg 2x">
  <img src="fallback.jpg" alt="Descripción">
</picture>
```

El scraper extraerá todas las imágenes.

### Ejemplo 2: Artículo con múltiples imágenes

```html
<article>
  <img src="thumbnail.jpg" alt="Portada" class="article-thumb">
  <figure>
    <img src="contenido.jpg" alt="Contenido principal">
    <figcaption>Leyenda</figcaption>
  </figure>
  <picture>
    <source srcset="responsive.jpg">
    <img src="fallback.jpg">
  </picture>
</article>
```

El scraper capturará todas, manteniendo información del contenedor padre (`<article>`, `<figure>`, `<picture>`).

### Ejemplo 3: Lazy loading

```html
<img src="placeholder.jpg" data-src="imagen-real.jpg" loading="lazy" alt="Lazy loaded">
```

El scraper intenta extraer del atributo `data-src`.

## Análisis de Resultados

### CSV principal (images_metadata.csv)

Perfecta para:
- Importar en Excel/Sheets
- Análisis en SQL
- Procesamiento con pandas

### JSON (images_metadata.json)

Perfecta para:
- Procesamiento programático
- APIs
- Almacenamiento en bases de datos NoSQL

### summary.json

Contiene estadísticas generales:
```json
{
  "timestamp": "2024-04-22T15:30:45.123456",
  "total_images": 150,
  "total_failed": 5,
  "total_unique_domains": 3,
  "average_image_size_kb": 324.5,
  "image_formats": ["image/jpeg", "image/png", "image/webp"],
  "domains": ["ejemplo.com", "cdn.ejemplo.com", "assets.ejemplo.com"]
}
```

## Troubleshooting

### Las imágenes descargadas son muy pocas

1. Verifica que el sitio **no esté bloqueando bots**
2. Aumenta `--max-pages`
3. Revisa el User-Agent (algunos sitios lo requieren específico)

### Error: `No module named 'requests'`

```bash
pip install requests beautifulsoup4 pillow
```

### El scraper es muy lento

- Aumenta `--delay` solo si es necesario (respeta el servidor)
- Reduce `--max-pages` para pruebas rápidas
- El delay por defecto (1s) es razonable

### Imágenes rotas o inválidas

El scraper automáticamente:
- Valida que sean imágenes reales (no archivos HTML)
- Verifica Content-Type correcto
- Detecta imágenes corruptas
- Registra fallos en el log

### Faltan imágenes del CSS o JavaScript

Este scraper solo extrae imágenes del HTML. Para imágenes cargadas dinámicamente, necesitarías:
- Usar Selenium o Playwright
- O permitir que JavaScript se ejecute (más complejo)

## Casos de Uso

### 1. Investigación de sitios de noticias

Scrapea artículos y recopila todas las imágenes con contexto:

```bash
python advanced_image_scraper.py \
    --url https://noticias.com \
    --max-pages 50 \
    --delay 2.0 \
    --output-dir ./noticias_dataset
```

Resultado: Dataset completo de imágenes con metadatos de origen, títulos, fechas, etc.

### 2. Análisis de presencia de marca

Busca logos y branding en sitios:

```bash
python advanced_image_scraper.py \
    --url https://competidor.com \
    --max-pages 20
```

Analiza cuáles son las imágenes más comunes, tamaños, formato.

### 3. Auditoría de accesibilidad

Identifica imágenes sin ALT text:

```python
with open('metadata/images_metadata.csv') as f:
    for row in csv.DictReader(f):
        if not row['alt_text']:
            print(f"Sin ALT: {row['image_url']}")
```

## Notas Importantes

1. **Respeta robots.txt**: Este scraper es educativo. Verifica siempre que puedas scrapear un sitio.

2. **Rate limiting**: El delay de 1 segundo es razonable. Aumenta si es necesario.

3. **Tamaño del storage**: Según la cantidad de imágenes, puede ocupar bastante espacio. Monitorea.

4. **Metadatos completos**: Cada imagen incluye toda la información para investigación y análisis posterior.

5. **Sin SVG**: Como solicitaste, excluye completamente los SVG.

## Ejemplos Avanzados

### Procesar metadatos con pandas

```python
import pandas as pd

df = pd.read_csv('metadata/images_metadata.csv')

# Imágenes más grandes
print(df.nlargest(10, 'file_size_kb')[['filename', 'file_size_kb', 'alt_text']])

# Distribución de tipos
print(df['image_type'].value_counts())

# Imágenes sin título
print(f"Sin título: {df[df['title'].isna() | (df['title'] == '')].shape[0]}")
```

### Exportar para otro procesamiento

```python
import json

# Agrupar por dominio para procesamiento
with open('metadata/images_metadata.json') as f:
    images = json.load(f)

by_domain = {}
for img in images:
    domain = img['source_domain']
    if domain not in by_domain:
        by_domain[domain] = []
    by_domain[domain].append(img)

for domain, imgs in by_domain.items():
    print(f"{domain}: {len(imgs)} imágenes")
```

---

¡Listo! Ahora tienes un scraper profesional de imágenes con metadatos completos. 🚀
