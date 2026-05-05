**APIs y rutas recomendadas para sitios de arte (resumen rápido)**

Objetivo: identificar APIs públicas o rutas fiables para obtener imágenes públicas de los sitios listados en `websites-list.csv`, y dar pasos prácticos para empezar (prioridad: usar APIs oficiales cuando sea posible).

- **Flickr**
  - API: Sí (https://www.flickr.com/services/api/). Requiere `api_key` (gratuito tras registro).
  - Qué permite: búsqueda por tags/usuario, obtener metadatos y construir URLs de las imágenes públicas.
  - Ejemplo mínimo (Python + requests):
    - Llamada: `https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=API_KEY&text=art&per_page=50&format=json&nojsoncallback=1`
    - Construir URL de imagen: `https://live.staticflickr.com/{server-id}/{id}_{secret}_{size-suffix}.jpg`.
  - Recomendación: empezar por Flickr para imágenes públicas y metadatos.

- **Behance (Adobe)**
  - API: Sí (https://www.behance.net/dev). Requiere `client_id` (registro Adobe).
  - Qué permite: listar proyectos, acceder a imágenes de proyectos y metadatos.
  - Recomendación: útil para portfolios; registrar app y usar API oficial.

- **DeviantArt**
  - API: Sí (https://www.deviantart.com/developers/). Usa OAuth2.
  - Qué permite: búsqueda, detalles de obras, descargas cuando están públicas.
  - Recomendación: requiere flujo OAuth para mayor volumen; para pruebas puntuales se puede scrapear páginas públicas respetando robots.txt.

- **Dribbble**
  - API: Sí (https://developer.dribbble.com/). Requiere token OAuth.
  - Qué permite: listar shots, imágenes y metadatos.
  - Recomendación: buena para diseño/ilustración moderna; obtener token para integraciones.

- **ArtStation**
  - API pública formal: limitada; sin embargo, ArtStation expone JSON embebido en páginas de proyecto y endpoints públicos no documentados (por ejemplo, `/projects/{slug}.json` o listados por usuario). No hay un OAuth oficial fácil.
  - Qué permite: muchas imágenes en portfolios; puede extraerse con requests a endpoints JSON o scraping de la página.
  - Recomendación: usar endpoints JSON cuando existan; respetar robots y límites.

- **Pixiv**
  - API: no pública para integraciones simples; existe API interna y wrappers no oficiales que requieren autenticación y manejo anti-bots.
  - Qué permite: todo el catálogo pero exige login y, en muchos casos, medidas anti-scraping.
  - Recomendación: evitar al principio si solo quieres imágenes públicas; buscar alternativas como ArtStation/DeviantArt/Behance.

- **Flickr (otra nota)**
  - Por volumen y facilidad de uso, Flickr es la mejor opción para empezar: fotos públicas, metadatos, límites claros y documentación.

- **Pinterest**
  - API: sí pero limitada y sometida a revisión (normalmente acceso restringido para partners).
  - Qué permite: pines y recursos visuales, pero acceso programático puede ser difícil.
  - Recomendación: preferir scraping respetuoso o usar fuentes alternativas.

- **Tumblr**
  - API: sí (https://www.tumblr.com/docs/en/api/v2). Permite acceder a posts y media; requiere key (API key simple disponible).
  - Recomendación: válido para blogs con imágenes públicas.

- **Reddit**
  - API: sí (https://www.reddit.com/dev/api/). Además muchos subreddits exponen JSON público en `https://www.reddit.com/r/<subreddit>.json` (lectura sin OAuth si se usa User-Agent adecuado y ritmo bajo).
  - Recomendación: útil para colecciones de imágenes públicas (subreddits de arte, ilustración, etc.).


Buenas prácticas y pasos recomendados para empezar (imágenes públicas):

- Priorizar APIs oficiales: Flickr, Behance, DeviantArt, Dribbble y Tumblr son los primeros objetivos.
- Pedir y almacenar claves (`api_key` / `client_id`) en variables de entorno o un `secrets.json` fuera del repo.
- Implementar peticiones respetuosas: respetar `robots.txt`, `Retry-After`, y límites de tasa de cada API.
- Para cada API, almacenar: URL original, imagen final, tamaño, ancho/alto, título, autor, licencia si está disponible.
- Evitar scraping masivo en sitios que requieren login o tienen medidas anti-bot (Pixiv, Pinterest en parte).

Snippets rápidos

- Ejemplo Python (Flickr) — obtener listados JSON:

```python
import os
import requests

API_KEY = os.getenv('FLICKR_API_KEY')
url = 'https://api.flickr.com/services/rest/'
params = {
    'method': 'flickr.photos.search',
    'api_key': API_KEY,
    'text': 'art',
    'per_page': 50,
    'format': 'json',
    'nojsoncallback': 1,
}
resp = requests.get(url, params=params, timeout=10)
data = resp.json()
for p in data.get('photos', {}).get('photo', []):
    server = p['server']; pid = p['id']; sec = p['secret']
    img_url = f'https://live.staticflickr.com/{server}/{pid}_{sec}_z.jpg'
    print(img_url)
```

- Ejemplo rápido (Reddit lectura pública):

```bash
curl -A 'my-agent' 'https://www.reddit.com/r/Art.json?limit=25'
```

Siguientes acciones que puedo hacer por ti ahora:

- 1) Ejecutar la `quick_art_probe.py` para todas las URLs del CSV y generar ranking automático.
- 2) Implementar un cliente minimal para `Flickr` (ejecutable) que descargue N imágenes públicas y sus metadatos.
- 3) Generar un script de onboarding que crea las variables de entorno y muestra cómo obtener claves para Flickr/Behance/DeviantArt.

Indica qué opción prefieres y lo hago (recomiendo empezar por Flickr: opción 2). 
