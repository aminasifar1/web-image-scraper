# 🚀 GUÍA RÁPIDA - Advanced Image Scraper

## Inicio Rápido (2 minutos)

### 1. Instalar dependencias
```bash
pip install -r requirements_scraper.txt
```

### 2. Ejecutar scraper
```bash
python advanced_image_scraper.py --url https://ejemplo.com
```

### 3. Ver resultados
```bash
ls -la image_scraper_results/
cat image_scraper_results/metadata/images_metadata.csv
```

---

## 📚 Archivos Principales

| Archivo | Descripción |
|---------|------------|
| **advanced_image_scraper.py** | 🔴 Scraper principal (el archivo que necesitas ejecutar) |
| **requirements_scraper.txt** | Dependencias necesarias |
| **examples.py** | 6 ejemplos educativos |
| **README_SCRAPER.md** | Documentación completa |
| **ADVANCED_SCRAPER_USAGE.md** | Guía detallada de uso |
| **QUICK_START.md** | Este archivo (guía rápida) |

---

## 🎯 Ejemplos por Caso de Uso

### Noticias (BBC, CNN, El País, etc.)
```bash
python advanced_image_scraper.py \
    --url https://www.bbc.com/news \
    --max-pages 10 \
    --delay 2.0
```

### Blog o Sitio Pequeño
```bash
python advanced_image_scraper.py --url https://mi-blog.com
```

### Múltiples Páginas (Requiere espera)
```bash
python advanced_image_scraper.py \
    --url https://ejemplo.com \
    --max-pages 20 \
    --delay 1.5 \
    --output-dir ./mis_resultados
```

---

## 📊 Analizar Resultados

### Ver estadísticas generales
```bash
python -c "
import json
with open('image_scraper_results/metadata/summary.json') as f:
    stats = json.load(f)
    print(f'Total imágenes: {stats[\"total_images\"]}')
    print(f'Dominios: {stats[\"total_unique_domains\"]}')
    print(f'Tamaño promedio: {stats[\"average_image_size_kb\"]:.1f} KB')
"
```

### Contar imágenes sin ALT (accesibilidad)
```bash
python -c "
import csv
with open('image_scraper_results/metadata/images_metadata.csv') as f:
    reader = csv.DictReader(f)
    no_alt = sum(1 for row in reader if not row.get('alt_text', ''))
    print(f'Imágenes sin ALT: {no_alt}')
"
```

### Listar imágenes más grandes
```bash
python -c "
import csv
with open('image_scraper_results/metadata/images_metadata.csv') as f:
    rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: float(x['file_size_kb']), reverse=True)
    for row in rows[:5]:
        print(f\"{row['filename']}: {float(row['file_size_kb']):.1f} KB\")
"
```

---

## ✅ Lo que Incluye

✅ Extrae de estructuras complejas (article, picture, source, video)  
✅ **Excluye .svg** automáticamente  
✅ Elimina duplicados (URL + hash)  
✅ Valida calidad (dimensiones, tamaño, formato)  
✅ Genera metadatos en CSV y JSON  
✅ Label de dominio de origen  
✅ Título, ALT, tipo MIME, fecha  
✅ Información HTML (clases, ID, padre)  
✅ Respetuoso con servidores (configurable)  

---

## 🔍 Filtros de Calidad Automáticos

| Criterio | Mínimo | Máximo |
|----------|--------|--------|
| Ancho | 100 px | 8000 px |
| Alto | 100 px | 8000 px |
| Área | 10K px² | - |
| Archivo | 5 KB | 10 MB |
| Formato | JPG, PNG, WEBP, GIF, BMP, TIFF | **NO SVG** |

---

## 📁 Estructura de Salida

```
image_scraper_results/
├── images/                    # Imágenes descargadas
│   ├── abc123.jpg
│   ├── def456.png
│   └── ...
│
└── metadata/                  # Metadatos
    ├── images_metadata.csv    # Todos los datos en CSV
    ├── images_metadata.json   # Todos los datos en JSON
    ├── images_ejemplo.csv    # Por dominio
    └── summary.json           # Estadísticas
```

---

## 🛠️ Opciones de Línea de Comandos

```
--url URL              Sitio a scrapear (REQUERIDO)
--output-dir PATH     Directorio de salida
--max-pages N         Máximo de páginas
--delay SEGUNDOS      Espera entre peticiones
```

**Ejemplo completo:**
```bash
python advanced_image_scraper.py \
    --url https://www.ejemplo.com \
    --output-dir ./mi_dataset \
    --max-pages 5 \
    --delay 2.0
```

---

## 🐛 Problemas Comunes

### "ModuleNotFoundError: No module named 'requests'"
```bash
pip install requests beautifulsoup4 pillow
```

### Descarga lenta
Aumenta delay conservadoramente:
```bash
--delay 0.5  # Más rápido (puede ser bloqueado)
--delay 2.0  # Más lento (más seguro)
```

### Pocas imágenes descargadas
```bash
# Intenta con más páginas
--max-pages 10
```

### El sitio bloquea el bot
- Verifica robots.txt
- Aumenta el delay
- Algunos sitios requieren User-Agent específico

---

## 🎓 Ejemplos Educativos

```bash
python examples.py
```

Incluye:
1. Scraping básico
2. Análisis de metadatos
3. Filtrar por criterios
4. Análisis de calidad
5. Reporte de accesibilidad
6. Scraping de múltiples páginas

---

## 📈 Casos Reales

### Investigar sitio de noticias
```bash
python advanced_image_scraper.py \
    --url https://www.theguardian.com \
    --max-pages 50 \
    --delay 2.0
```
→ Dataset de 500+ imágenes con contexto

### Auditoría de accesibilidad
Busca imágenes sin ALT text en CSV

### Dataset para ML
Exporta a JSON para procesamiento

### Análisis competitivo
Examina imágenes usadas por competidores

---

## 📞 Soporte Rápido

**¿Cuántas imágenes puede descargar?**  
Depende de:
- Cantidad de páginas (`--max-pages`)
- Cantidad de imágenes por página
- Filtros de calidad activados

**¿Cuánto espacio necesita?**  
~100-300 MB por 1000 imágenes (promedio)

**¿Es seguro/legal?**  
✓ Respeta robots.txt  
✓ Configurable (delay)  
⚠️ Verifica términos de servicio del sitio  

---

## 🚀 Siguiente Paso

Lee la documentación completa:
```bash
less README_SCRAPER.md
less ADVANCED_SCRAPER_USAGE.md
```

¡O comienza a scrapear!
```bash
python advanced_image_scraper.py --url https://ejemplo.com
```

---

**Status:** ✅ Listo para usar  
**Versión:** 1.0  
**Última actualización:** 2024-04-22  
**Pruebas:** 4/4 ✅
