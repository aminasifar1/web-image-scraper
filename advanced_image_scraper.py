#!/usr/bin/env python3
"""
Advanced Web Image Scraper
Extrae imágenes de sitios web con:
- Soporte para estructuras HTML complejas (article, source, picture, etc.)
- Exclusión de archivos SVG
- Deduplicación automática
- Filtros de calidad de imagen
- Metadatos detallados incluyendo origen, título, ALT, tipo, fecha, etc.

Uso:
    python advanced_image_scraper.py --url https://ejemplo.com --output-dir ./resultados
"""

import argparse
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO


def _normalize_context_text(value: str) -> str:
    """Normaliza texto para comparar contexto semántico (title/alt)."""
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _extract_asset_signature(image_name: str, image_url: str) -> str:
    """Extrae una firma estable del recurso para agrupar variantes responsive."""
    raw_name = image_name or Path(urlparse(image_url).path).name
    if not raw_name:
        return ""

    stem = Path(raw_name).stem.lower()

    # Caso típico en lavanguardia: 69e8eeafc28ba.r_d.1844-1263[-987]
    stem = re.sub(r'\.r_d\.\d+-\d+(?:-\d+)?$', '', stem)

    # Patrones comunes de tamaños al final: *_290x300, -290x300, _449_221
    stem = re.sub(r'([_-])\d{2,5}x\d{2,5}$', '', stem)
    stem = re.sub(r'([_-])\d{2,5}([_-])\d{2,5}$', '', stem)

    return stem.strip('._-')


def canonicalize_image_url(image_url: str) -> str:
    """Normaliza una URL de imagen para detectar variantes del mismo asset.

    Reglas:
    - Ignora segmentos de path del tipo `image_WIDTH_HEIGHT`.
        - Ignora segmentos de preset/filtro del CMS (p.ej. `content_image_mobile_filter`).
    - Conserva la ruta real de subida (por ejemplo `files/fp/uploads/...`).
    - Usa el nombre del archivo sin extensión.
    - Elimina un sufijo numérico final como `-1051` solo si está al final.
        - Para patrones `*.r_d.*` conserva el identificador base previo a `.r_d.`
            para agrupar crops/variantes del mismo asset.
        - Mantiene el resto del nombre intacto.
    """
    if not image_url:
        return ""

    parsed = urlparse(image_url.strip())
    if parsed.scheme not in {'http', 'https'}:
        return ""

    normalized_parts: List[str] = []
    previous_part = None
    for part in parsed.path.split('/'):
        if not part:
            continue
        if re.fullmatch(r'image_\d+_\d+', part):
            continue
        if re.fullmatch(r'content_image_[a-z0-9_]+', part):
            continue
        if part.startswith('content_image_'):
            continue
        if part == previous_part:
            continue
        normalized_parts.append(part)
        previous_part = part

    if not normalized_parts:
        return f"{parsed.netloc.lower()}|"

    filename = normalized_parts[-1]
    stem = Path(filename).stem
    # Variantes típicas de lavanguardia: <id>.r_d.<crop-o-dimensiones>
    # Se conserva solo el identificador base del asset para unificar variantes.
    if '.r_d.' in stem:
        stem = stem.split('.r_d.')[0]
    # Solo eliminar el último segmento numérico cuando el nombre ya contiene
    # una cola numérica compuesta por varios grupos (por ejemplo `...-2084-1311-1051`).
    # Así preservamos identificadores base como `...-2084-1311`.
    if re.search(r'(?:-\d+){2,}$', stem):
        stem = re.sub(r'-\d+$', '', stem)
    normalized_parts[-1] = stem

    normalized_path = '/'.join(normalized_parts)
    return f"{parsed.netloc.lower()}|{normalized_path}"


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Extensiones de imagen permitidas (EXCLUYE .svg)
ALLOWED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'
}

# Tamaños mínimos de imagen para calidad
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_AREA = 10000  # píxeles
MIN_IMAGE_SIZE_KB = 5  # 5 KB

# Tamaños máximos para evitar imágenes enormes
MAX_IMAGE_WIDTH = 8000
MAX_IMAGE_HEIGHT = 8000
MAX_IMAGE_SIZE_MB = 10  # 10 MB

# Límite de seguridad de páginas para evitar crawls descontrolados
SAFE_MAX_PAGES = 300

# Configuración de deduplicación perceptual
DEFAULT_PERCEPTUAL_HASH_SIZE = 8
DEFAULT_PERCEPTUAL_THRESHOLD = 6

# Solo palabras clave para contenido realmente inapropiado (ilegal/NSFW)
# Se buscan solo en URL y ALT text, no en clases/IDs
BLOCKED_IMAGE_KEYWORDS = {
    'nsfw', 'porn', 'adult', 'nud', 'nude', 'sex',
    'gore', 'graphic', 'violence', 'bloody', 'illegal', 'explicit'
}

# Prioridades de crawling (menor score = mayor prioridad)
# Enfoque genérico para cualquier tipo de web: no sesgar a un vertical concreto.
HIGH_PRIORITY_PATH_HINTS = (
    '/article/', '/post/', '/blog/', '/news/', '/story/',
    '/product/', '/item/', '/detail/', '/media/', '/gallery/'
)

LOW_PRIORITY_PATH_HINTS = (
    '/login', '/signup', '/register', '/logout', '/account', '/profile',
    '/terms', '/privacy', '/cookies', '/legal', '/status', '/health', '/api-docs'
)

# Indicadores de recursos no editoriales (iconos/badges/sprites)
ICON_LIKE_KEYWORDS = {
    'icon', 'favicon', 'sprite', 'badge', 'logo', 'emoji', 'avatar',
    'app-store', 'google-play', 'play-badge', 'social-icon'
}

GENERIC_ICON_ALT_TEXTS = {
    'icon', 'logo', 'badge', 'menu', 'search', 'close', 'next', 'previous',
    'facebook', 'instagram', 'x', 'twitter', 'youtube', 'tiktok'
}


@dataclass
class ImageMetadata:
    """Metadatos completos de una imagen"""
    # Identificadores
    image_id: str
    image_url: str
    source_domain: str
    source_url: str
    
    # Información de contenido
    title: str = ""
    alt_text: str = ""
    image_type: str = ""  # MIME type
    filename: str = ""
    
    # Calidad y dimensiones
    width: int = 0
    height: int = 0
    file_size_kb: float = 0.0
    image_hash: str = ""
    
    # Contexto HTML
    html_tag: str = ""  # img, source, etc.
    parent_tag: str = ""  # article, picture, etc.
    classes: str = ""
    element_id: str = ""
    
    # Fecha de la imagen
    image_timestamp: str = ""  # Fecha EXIF o Last-Modified
    image_date: str = ""  # Solo la fecha (YYYY-MM-DD)
    output_format: str = ""  # Formato final en disco
    
    # Información adicional
    image_name_from_url: str = ""
    canonical_image_key: str = ""
    perceptual_hash: str = ""
    srcset: str = ""  # Para fuentes responsive
    loading_strategy: str = ""  # lazy, eager, etc.
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para CSV"""
        return asdict(self)


class AdvancedImageScraper:
    """Scraper avanzado de imágenes con soporte para múltiples formatos HTML"""
    
    def __init__(
        self,
        output_dir: str = "./image_scraper_results",
        delay: float = 1.0,
        convert_to_png: bool = False,
        perceptual_threshold: int = DEFAULT_PERCEPTUAL_THRESHOLD,
        perceptual_hash_size: int = DEFAULT_PERCEPTUAL_HASH_SIZE,
    ):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.convert_to_png = convert_to_png
        self.perceptual_threshold = max(0, int(perceptual_threshold))
        self.perceptual_hash_size = max(4, int(perceptual_hash_size))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Crear directorios
        self._create_directories()
        
        # Para tracking
        self.downloaded_images: List[ImageMetadata] = []
        self.failed_urls: List[str] = []
        self.duplicate_events: List[Dict[str, str]] = []
        self.seen_image_urls: Set[str] = set()
        self.seen_canonical_keys: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.seen_perceptual_hashes: List[Tuple[int, str]] = []
        self.seen_pages: Set[str] = set()

    def _register_duplicate(self, image_url: str, reason: str, matched_with: str = "", extra: Optional[Dict] = None):
        """Registra duplicados con motivo para auditoría del proceso."""
        entry: Dict[str, str] = {
            'image_url': image_url,
            'reason': reason,
        }
        if matched_with:
            entry['matched_with'] = matched_with
        if extra:
            for key, value in extra.items():
                entry[str(key)] = str(value)
        self.duplicate_events.append(entry)
    
    def _create_directories(self):
        """Crea la estructura de directorios necesaria"""
        dirs = {
            'images': self.output_dir / 'images_lavanguardia',
            'metadata': self.output_dir / 'metadata',
            'logs': self.output_dir / 'logs'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _extract_images_from_html(self, html_content: str, page_url: str) -> List[Dict]:
        """
        Extrae imágenes de múltiples fuentes HTML:
        - <img> tags
        - <picture> -> <source> con srcset
        - <article> -> <img>
        - <video> con <source>
        - <figure> -> <img>
        - data attributes lazy loading
        """
        images = []
        soup = BeautifulSoup(html_content, 'html.parser')
        processed_img_nodes: Set[int] = set()
        
        # 1. Extraer de <img> tags
        for img_tag in soup.find_all('img'):
            processed_img_nodes.add(id(img_tag))
            image_data = self._extract_from_img_tag(img_tag, page_url)
            if image_data:
                images.append(image_data)

        # 1.b Extraer imágenes dentro de <noscript> (común en lazy-loading)
        for noscript_tag in soup.find_all('noscript'):
            noscript_html = noscript_tag.decode_contents() or ''
            if not noscript_html.strip():
                continue
            nested_soup = BeautifulSoup(noscript_html, 'html.parser')
            for img_tag in nested_soup.find_all('img'):
                image_data = self._extract_from_img_tag(img_tag, page_url)
                if image_data:
                    images.append(image_data)
        
        # 2. Extraer de <picture> -> <source>
        for picture_tag in soup.find_all('picture'):
            image_data = self._extract_from_picture_tag(picture_tag, page_url)
            if image_data:
                images.append(image_data)
        
        # 3. Extraer de <video> -> <source>
        for video_tag in soup.find_all('video'):
            poster = video_tag.get('poster')
            if poster and self._is_valid_image_url(poster):
                image_data = self._extract_from_element(
                    poster, 'source', 'video', page_url, 
                    alt_text=f"Video poster: {video_tag.get('title', '')}"
                )
                if image_data:
                    images.append(image_data)
        
        # 4. Extraer de contenedores semánticos y wrappers con clase
        # Muchas páginas de arte usan estructuras como <div class="..."><img ...></div>
        # por lo que recorremos también el ancestro con clase más cercano para
        # registrar mejor el contexto de la imagen.
        for container_tag in soup.find_all(['article', 'figure', 'section', 'div', 'a']):
            for img_tag in container_tag.find_all('img', recursive=True):
                if id(img_tag) in processed_img_nodes:
                    continue
                # Evitar duplicados
                src = self._get_preferred_image_url(img_tag)
                if src and self._get_image_hash_from_url(src) not in {
                    self._get_image_hash_from_url(img.get('image_url', ''))
                    for img in images
                }:
                    image_data = self._extract_from_img_tag(img_tag, page_url, parent_tag=container_tag.name)
                    if image_data:
                        images.append(image_data)
        
        return images
    
    def _extract_from_img_tag(self, img_tag, page_url: str, parent_tag: str = None) -> Optional[Dict]:
        """Extrae información de un tag <img>"""
        # Intentar obtener URL de múltiples atributos (data-*, srcset, src)
        # y probarlas en orden para no perder imágenes válidas que estén solo en src.
        image_candidates = self._get_image_url_candidates(img_tag)
        if not image_candidates:
            return None

        resolved_parent_tag = parent_tag or self._find_container_tag(img_tag) or img_tag.parent.name
        resolved_classes = self._merge_classes(img_tag.get('class', []), self._container_classes(img_tag))
        resolved_alt_text, resolved_title = self._resolve_image_text_context(img_tag)

        for image_url in image_candidates:
            image_data = self._extract_from_element(
                image_url=image_url,
                html_tag='img',
                parent_tag=resolved_parent_tag,
                page_url=page_url,
                alt_text=resolved_alt_text,
                title=resolved_title,
                classes=resolved_classes,
                element_id=img_tag.get('id', ''),
                srcset=img_tag.get('srcset', ''),
                loading=img_tag.get('loading', '')
            )
            if image_data:
                return image_data

        return None

    def _resolve_image_text_context(self, img_tag) -> Tuple[str, str]:
        """Resuelve ALT/TITLE con fallback contextual cuando ALT viene vacío."""
        raw_alt = (img_tag.get('alt') or '').strip()
        raw_title = (img_tag.get('title') or '').strip()

        if raw_alt and raw_title:
            return self._clean_metadata_text(raw_alt), self._clean_metadata_text(raw_title)

        context_candidates: List[str] = []

        # 1) Prioridad alta: figcaption cercano de la imagen.
        figure = img_tag.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption:
                context_candidates.append(figcaption.get_text(' ', strip=True))

        # 2) Titulares cercanos dentro del artículo/contendor.
        article = img_tag.find_parent('article')
        if article:
            for selector in ('h1', 'h2', 'h3'):
                node = article.find(selector)
                if node:
                    context_candidates.append(node.get_text(' ', strip=True))
            # Enlaces de contexto secundarios (subtítulos / relacionados).
            for anchor in article.find_all('a', limit=4):
                text = anchor.get_text(' ', strip=True)
                if text:
                    context_candidates.append(text)
            summary = article.find('p')
            if summary:
                context_candidates.append(summary.get_text(' ', strip=True))

        # 3) Último recurso: texto del padre inmediato.
        parent = getattr(img_tag, 'parent', None)
        if parent:
            context_candidates.append(parent.get_text(' ', strip=True))

        normalized_candidates: List[str] = []
        for candidate in context_candidates:
            cleaned = self._clean_metadata_text(candidate)
            if cleaned and cleaned not in normalized_candidates:
                normalized_candidates.append(cleaned)

        if not raw_alt and normalized_candidates:
            raw_alt = normalized_candidates[0]
        if not raw_title and len(normalized_candidates) > 1:
            raw_title = normalized_candidates[1]

        return self._clean_metadata_text(raw_alt), self._clean_metadata_text(raw_title)

    def _clean_metadata_text(self, value: str, max_len: int = 280) -> str:
        """Normaliza texto para metadatos evitando ruido y cadenas excesivas."""
        if not value:
            return ''
        normalized = re.sub(r'\s+', ' ', value).strip()
        if len(normalized) > max_len:
            normalized = normalized[:max_len].rstrip() + '...'
        return normalized

    def _get_preferred_image_url(self, img_tag) -> Optional[str]:
        """Devuelve la mejor URL disponible del <img>, priorizando variantes de alta resolución."""
        candidates = self._get_image_url_candidates(img_tag)
        return candidates[0] if candidates else None

    def _get_image_url_candidates(self, img_tag) -> List[str]:
        """Devuelve URLs candidatas ordenadas para un <img> (alta resolución -> fallback)."""
        candidates: List[str] = []

        def add_candidate(value: Optional[str]) -> None:
            if value and value not in candidates:
                candidates.append(value)

        for attribute_name in (
            'data-large',
            'data-full',
            'data-original',
            'data-src',
            'data-lazy-src',
        ):
            candidate = img_tag.get(attribute_name)
            add_candidate(candidate)

        data_srcset = img_tag.get('data-srcset', '')
        best_data_srcset_url = self._select_best_srcset_url(data_srcset)
        add_candidate(best_data_srcset_url)

        srcset = img_tag.get('srcset', '')
        best_srcset_url = self._select_best_srcset_url(srcset)
        add_candidate(best_srcset_url)

        src = img_tag.get('src')
        add_candidate(src)

        return candidates

    def _select_best_srcset_url(self, srcset: str) -> Optional[str]:
        """Selecciona la URL de mayor resolución desde un `srcset`."""
        if not srcset:
            return None

        best_url = None
        best_score = -1.0

        for candidate in re.split(r',\s+', srcset.strip()):
            candidate = candidate.strip()
            if not candidate:
                continue
            parts = candidate.split()
            url = parts[0]
            score = 0.0
            if len(parts) > 1:
                descriptor = parts[-1].lower()
                if descriptor.endswith('w'):
                    try:
                        score = float(descriptor[:-1])
                    except ValueError:
                        score = 0.0
                elif descriptor.endswith('x'):
                    try:
                        score = float(descriptor[:-1]) * 1000.0
                    except ValueError:
                        score = 0.0
            if score >= best_score:
                best_score = score
                best_url = url

        return best_url

    def _find_container_tag(self, img_tag):
        """Busca el contenedor semántico o con clase más cercano para un <img>."""
        for ancestor in img_tag.parents:
            if getattr(ancestor, 'name', None) == 'a':
                return 'a'

        for ancestor in img_tag.parents:
            if getattr(ancestor, 'name', None) not in {'article', 'figure', 'section', 'div', 'a'}:
                continue
            if ancestor.get('class'):
                return ancestor.name
        return None

    def _container_classes(self, img_tag) -> List[str]:
        """Devuelve las clases de wrappers relevantes para enriquecer el contexto."""
        collected: List[str] = []
        for ancestor in img_tag.parents:
            if getattr(ancestor, 'name', None) not in {'article', 'figure', 'section', 'div', 'a'}:
                continue
            classes = ancestor.get('class') or []
            if classes:
                for class_name in classes:
                    if class_name and class_name not in collected:
                        collected.append(class_name)
        return collected

    def _merge_classes(self, image_classes: List[str], container_classes: List[str]) -> str:
        """Combina clases del <img> y del contenedor evitando duplicados."""
        merged: List[str] = []
        for class_name in list(image_classes) + list(container_classes):
            if class_name and class_name not in merged:
                merged.append(class_name)
        return ' '.join(merged)
    
    def _extract_from_picture_tag(self, picture_tag, page_url: str) -> Optional[Dict]:
        """Extrae información de tags <picture> -> <source>"""
        images = []
        
        # Buscar <source> tags con srcset
        for source_tag in picture_tag.find_all('source'):
            srcset = source_tag.get('srcset', '')
            if srcset:
                # Obtener la primera URL del srcset
                image_url = srcset.split(',')[0].strip().split(' ')[0]
                if image_url:
                    image_data = self._extract_from_element(
                        image_url=image_url,
                        html_tag='source',
                        parent_tag='picture',
                        page_url=page_url,
                        srcset=srcset,
                        media_query=source_tag.get('media', '')
                    )
                    if image_data:
                        images.append(image_data)
        
        # También extraer <img> dentro de <picture>
        img_tag = picture_tag.find('img')
        if img_tag:
            image_data = self._extract_from_img_tag(img_tag, page_url, parent_tag='picture')
            if image_data:
                images.append(image_data)
        
        return images[0] if images else None
    
    def _extract_from_element(
        self,
        image_url: str,
        html_tag: str,
        parent_tag: str,
        page_url: str,
        alt_text: str = "",
        title: str = "",
        classes: str = "",
        element_id: str = "",
        srcset: str = "",
        loading: str = "",
        media_query: str = ""
    ) -> Optional[Dict]:
        """Extrae y valida información de una fuente de imagen"""
        
        # Validar y normalizar URL
        image_url = image_url.strip()
        if not image_url:
            return None
        
        # Hacer URL absoluta
        absolute_url = urljoin(page_url, image_url)
        
        # Validar formato
        if not self._is_valid_image_url(absolute_url):
            return None

        if self._should_skip_image_candidate(absolute_url, alt_text, title, classes, element_id, html_tag, parent_tag, srcset, loading):
            return None

        # Evitar duplicados exactos por URL
        if absolute_url in self.seen_image_urls:
            self._register_duplicate(absolute_url, 'exact_url_duplicate', matched_with=absolute_url)
            return None

        # Canonicalizar la URL para agrupar variantes servidas por el CMS.
        # Esta clave ignora tamaños embebidos en el path y segmentos de preset,
        # para que dos variantes del mismo asset compartan huella lógica.
        canonical_image_key = canonicalize_image_url(absolute_url)
        if canonical_image_key and canonical_image_key in self.seen_canonical_keys:
            logger.debug(f"Imagen duplicada por canonical_image_key: {absolute_url}")
            self._register_duplicate(absolute_url, 'canonical_duplicate', matched_with=canonical_image_key)
            return None
        
        self.seen_image_urls.add(absolute_url)
        if canonical_image_key:
            self.seen_canonical_keys.add(canonical_image_key)
        
        # Extraer información de URL
        image_name = Path(urlparse(absolute_url).path).name or "image"
        source_domain = urlparse(page_url).netloc
        
        return {
            'image_url': absolute_url,
            'page_url': page_url,
            'source_domain': source_domain,
            'html_tag': html_tag,
            'parent_tag': parent_tag,
            'alt_text': alt_text,
            'title': title,
            'classes': classes,
            'element_id': element_id,
            'srcset': srcset,
            'loading': loading,
            'media_query': media_query,
            'image_name': image_name,
            'canonical_image_key': canonical_image_key
        }

    def _should_skip_image_candidate(
        self,
        image_url: str,
        alt_text: str = "",
        title: str = "",
        classes: str = "",
        element_id: str = "",
        html_tag: str = "",
        parent_tag: str = "",
        srcset: str = "",
        loading: str = "",
    ) -> bool:
        # Solo buscar palabras bloqueadas en URL y alt_text
        # Ignorar clases CSS, IDs, tags y otros atributos técnicos
        content_to_check = (image_url + " " + alt_text).lower()
        if any(keyword in content_to_check for keyword in BLOCKED_IMAGE_KEYWORDS):
            return True

        # Evitar iconos/badges no editoriales, pero conservar imágenes con ALT
        # descriptivo (p.ej. obras en galerías de arte).
        if self._is_likely_non_content_icon(image_url, alt_text, title):
            return True

        return False

    def _is_likely_non_content_icon(self, image_url: str, alt_text: str = "", title: str = "") -> bool:
        """Heurística suave para descartar iconos sin perder contenido editorial."""
        normalized_url = (image_url or '').lower()
        normalized_alt = _normalize_context_text(alt_text or '')
        normalized_title = _normalize_context_text(title or '')
        description = (normalized_alt + ' ' + normalized_title).strip()

        # Si hay descripción suficientemente rica, se considera contenido válido.
        if len(description) >= 20 and len(description.split()) >= 3:
            return False

        # ALT/TITLE genérico típico de icono.
        if description and description in GENERIC_ICON_ALT_TEXTS:
            return True

        # URL claramente de iconografía y sin descripción útil.
        if any(keyword in normalized_url for keyword in ICON_LIKE_KEYWORDS):
            return True

        return False
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Valida que la URL sea una imagen válida (no SVG)"""
        # Esquema válido
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https', 'data'}:
            return False
        
        # Obtener extensión
        path = parsed.path.lower()
        extension = Path(path).suffix.lower()
        
        # Excluir SVG
        if extension == '.svg':
            return False
        
        # Si tiene extensión, debe ser válida
        if extension and extension not in ALLOWED_IMAGE_EXTENSIONS:
            return False
        
        # Si no tiene extensión, probablemente sea válido (podría ser servido dinámicamente)
        return True

    def _discover_sitemap_urls(self, seed_url: str, allowed_domains: Set[str], limit: int = 5000) -> List[str]:
        """Descubre URLs internas a partir de sitemaps, evitando ciclos y recursión infinita."""
        parsed_seed = urlparse(seed_url)
        normalized_limit = max(0, int(limit))
        if normalized_limit == 0:
            return []

        sitemap_queue: List[str] = [
            f"{parsed_seed.scheme}://{parsed_seed.netloc}/sitemap.xml",
            f"{parsed_seed.scheme}://{parsed_seed.netloc}/sitemap_index.xml",
            f"{parsed_seed.scheme}://{parsed_seed.netloc}/sitemap-index.xml",
        ]
        visited_sitemaps: Set[str] = set()
        max_sitemaps_to_fetch = 1000

        discovered: List[str] = []
        seen_urls: Set[str] = set()

        def add_url(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized or normalized in seen_urls:
                return

            parsed = urlparse(normalized)
            if parsed.scheme not in {'http', 'https'}:
                return
            if allowed_domains and parsed.netloc.lower() not in allowed_domains:
                return

            path = (parsed.path or '').lower()
            if path.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.svg')):
                return

            seen_urls.add(normalized)
            discovered.append(normalized)

        def looks_like_sitemap(candidate: str) -> bool:
            parsed = urlparse(candidate)
            path = (parsed.path or '').lower()
            return path.endswith('.xml') or 'sitemap' in path

        while sitemap_queue and len(discovered) < normalized_limit and len(visited_sitemaps) < max_sitemaps_to_fetch:
            sitemap_url = sitemap_queue.pop(0).strip()
            if not sitemap_url or sitemap_url in visited_sitemaps:
                continue

            visited_sitemaps.add(sitemap_url)

            try:
                response = self.session.get(sitemap_url, timeout=10)
                if response.status_code != 200:
                    continue

                text = response.text.strip()
                if not text:
                    continue

                root = ET.fromstring(text)
                namespace = ''
                if root.tag.startswith('{'):
                    namespace = root.tag.split('}')[0].strip('{')

                loc_xpath = f'.//{{{namespace}}}loc' if namespace else './/loc'
                for loc in root.findall(loc_xpath):
                    loc_text = (loc.text or '').strip()
                    if not loc_text:
                        continue

                    if looks_like_sitemap(loc_text):
                        if loc_text not in visited_sitemaps:
                            sitemap_queue.append(loc_text)
                        continue

                    add_url(loc_text)
                    if len(discovered) >= normalized_limit:
                        break

            except Exception:
                continue

        if len(visited_sitemaps) >= max_sitemaps_to_fetch:
            logger.warning(
                f"Límite de seguridad de sitemaps alcanzado ({max_sitemaps_to_fetch}) para {seed_url}"
            )

        return discovered[:normalized_limit]

    def _extract_links_from_page(self, page_url: str, soup: BeautifulSoup) -> List[str]:
        """Extrae todos los enlaces internos de una página HTML"""
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue
            resolved = urljoin(page_url, href)
            parsed = urlparse(resolved)
            if parsed.scheme in {'http', 'https'}:
                links.append(resolved)
        return links

    def _normalize_internal_link(self, page_url: str, link_url: str, allowed_domains: Set[str]) -> Optional[str]:
        resolved = urljoin(page_url, link_url)
        parsed = urlparse(resolved)
        if parsed.scheme not in {'http', 'https'}:
            return None
        if allowed_domains and parsed.netloc.lower() not in allowed_domains:
            return None
        return resolved

    def _link_priority_score(self, link_url: str) -> int:
        """Calcula prioridad de crawling con heurística genérica y amplia cobertura."""
        parsed = urlparse(link_url)
        path = (parsed.path or '/').lower()
        score = 40

        # Penalización suave para rutas de sesión/legales/técnicas.
        # Se mantienen en cola (no se excluyen), para maximizar cobertura.
        for hint in LOW_PRIORITY_PATH_HINTS:
            if hint in path:
                score += 20
                break

        # Bonus leve para rutas de contenido habituales en múltiples tipos de sitios.
        for hint in HIGH_PRIORITY_PATH_HINTS:
            if hint in path:
                score -= 15
                break

        # Páginas internas suelen contener más recursos embebidos.
        depth = sum(1 for segment in path.split('/') if segment)
        if depth >= 3:
            score -= 10
        elif depth == 2:
            score -= 5
        elif depth == 0:
            score += 15

        # URLs con query suelen incluir listados/paginación/filtros con imágenes.
        if parsed.query:
            score -= 5

        return max(0, score)

    def _prioritize_to_visit_queue(self, to_visit: List[str], visited_urls: Set[str]) -> List[str]:
        """Ordena la cola para procesar primero páginas de mayor valor de contenido."""
        unique_pending = []
        seen_pending: Set[str] = set()
        for candidate in to_visit:
            if candidate in visited_urls or candidate in seen_pending:
                continue
            seen_pending.add(candidate)
            unique_pending.append(candidate)

        unique_pending.sort(key=lambda candidate: (self._link_priority_score(candidate), candidate))
        return unique_pending
    
    def _get_image_hash_from_url(self, url: str) -> str:
        """Calcula hash de la URL para deduplicación"""
        return hashlib.md5(url.encode()).hexdigest()

    def _compute_dhash(self, image_bytes: bytes) -> Optional[int]:
        """Calcula dHash perceptual para detectar imágenes visualmente similares."""
        try:
            hash_size = self.perceptual_hash_size
            with Image.open(BytesIO(image_bytes)) as image:
                gray = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
                pixels = list(gray.tobytes())

            bits = 0
            bit_index = 0
            for row in range(hash_size):
                row_offset = row * (hash_size + 1)
                for col in range(hash_size):
                    left = pixels[row_offset + col]
                    right = pixels[row_offset + col + 1]
                    if left > right:
                        bits |= (1 << bit_index)
                    bit_index += 1
            return bits
        except Exception as e:
            logger.debug(f"No se pudo calcular dHash: {e}")
            return None

    def _hamming_distance(self, hash_a: int, hash_b: int) -> int:
        """Distancia de Hamming entre dos perceptual hashes."""
        return (hash_a ^ hash_b).bit_count()

    def _find_perceptual_duplicate(self, candidate_hash: int) -> Optional[Tuple[str, int]]:
        """Busca duplicado perceptual por umbral configurable."""
        best_match: Optional[Tuple[str, int]] = None
        best_distance: Optional[int] = None
        for seen_hash, seen_url in self.seen_perceptual_hashes:
            distance = self._hamming_distance(candidate_hash, seen_hash)
            if distance <= self.perceptual_threshold:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = (seen_url, distance)
        return best_match
    
    def _download_image(self, image_data: Dict) -> Optional[Tuple[bytes, str, Dict]]:
        """Descarga una imagen y valida su calidad
        
        Returns:
            Tupla (contenido, content_type, headers) o None
        """
        try:
            response = self.session.get(
                image_data['image_url'],
                timeout=10,
                stream=True
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Error descargando {image_data['image_url']}: {e}")
            self.failed_urls.append(image_data['image_url'])
            return None
        
        # Validar content-type
        content_type = response.headers.get('Content-Type', '').lower()
        if content_type and not content_type.startswith('image/'):
            return None
        
        # Validar tamaño
        content_length = response.headers.get('Content-Length')
        if content_length:
            size_kb = int(content_length) / 1024
            if size_kb < MIN_IMAGE_SIZE_KB or size_kb > MAX_IMAGE_SIZE_MB * 1024:
                return None
        
        image_data['file_size_kb'] = len(response.content) / 1024
        
        return response.content, content_type, response.headers
    
    def _extract_image_date(self, image_bytes: bytes, response_headers: Dict = None) -> Tuple[str, str]:
        """Extrae la fecha de la imagen desde EXIF o headers HTTP
        
        Returns:
            Tupla (timestamp ISO, fecha YYYY-MM-DD)
        """
        image_date = ""
        image_timestamp = ""
        
        # 1. Intentar obtener de metadatos EXIF
        try:
            image = Image.open(BytesIO(image_bytes))
            exif_data = image._getexif() if hasattr(image, '_getexif') else None
            
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    # Buscar fecha en varios campos EXIF
                    if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        try:
                            # Formato típico EXIF: "2023:12:15 14:30:45"
                            dt = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            image_timestamp = dt.isoformat()
                            image_date = dt.strftime('%Y-%m-%d')
                            logger.debug(f"Fecha EXIF encontrada: {image_date}")
                            return image_timestamp, image_date
                        except:
                            continue
        except Exception as e:
            logger.debug(f"No se pudo leer EXIF: {e}")
        
        # 2. Intentar obtener del header Last-Modified del servidor
        if response_headers and not image_date:
            try:
                last_modified = response_headers.get('Last-Modified')
                if last_modified:
                    dt = parsedate_to_datetime(last_modified)
                    image_timestamp = dt.isoformat()
                    image_date = dt.strftime('%Y-%m-%d')
                    logger.debug(f"Fecha Last-Modified encontrada: {image_date}")
                    return image_timestamp, image_date
            except Exception as e:
                logger.debug(f"Error parseando Last-Modified: {e}")
        
        # 3. Si no hay fecha, retornar vacío
        logger.debug("No se encontró fecha EXIF ni Last-Modified")
        return "", ""
    
    def _validate_image_quality(self, image_bytes: bytes) -> Tuple[bool, int, int, str]:
        """Valida la calidad de la imagen"""
        try:
            image = Image.open(BytesIO(image_bytes))
            width, height = image.size
            format_type = image.format or 'UNKNOWN'
            
            # Validaciones de calidad
            checks = [
                ('width', width >= MIN_IMAGE_WIDTH),
                ('height', height >= MIN_IMAGE_HEIGHT),
                ('area', width * height >= MIN_IMAGE_AREA),
                ('max_width', width <= MAX_IMAGE_WIDTH),
                ('max_height', height <= MAX_IMAGE_HEIGHT),
            ]
            
            is_valid = all(check[1] for check in checks)
            
            return is_valid, width, height, format_type
        
        except Exception as e:
            logger.warning(f"Error validando calidad de imagen: {e}")
            return False, 0, 0, 'ERROR'
    
    def scrape(self, url: str, max_pages: int =10, crawl_site: bool = False) -> Dict:
        """
        Realiza el scraping de un sitio web
        
        Args:
            url: URL del sitio a scrapear
            max_pages: Número máximo de páginas a procesar
        
        Returns:
            Diccionario con estadísticas del proceso
        """
        logger.info(f"Iniciando scraping de {url}")
        
        visited_urls = set()
        allowed_domains = {urlparse(url).netloc.lower()}
        to_visit = [url]
        if crawl_site:
            sitemap_urls = self._discover_sitemap_urls(url, allowed_domains, limit=max_pages * 20)
            to_visit = sitemap_urls + to_visit
        to_visit = self._prioritize_to_visit_queue(to_visit, visited_urls)
        pages_processed = 0
        
        # Evitar crawls infinitos: si max_pages <= 0, aplicar límite de seguridad
        if max_pages <= 0:
            page_limit = SAFE_MAX_PAGES
            logger.warning(
                f"max_pages={max_pages} detectado; usando límite de seguridad de {SAFE_MAX_PAGES} páginas"
            )
        else:
            page_limit = min(max_pages, SAFE_MAX_PAGES)
            if max_pages > SAFE_MAX_PAGES:
                logger.warning(
                    f"max_pages={max_pages} excede el límite de seguridad; se usará {SAFE_MAX_PAGES}"
                )

        while to_visit and pages_processed < page_limit:
            current_url = to_visit.pop(0)
            
            if current_url in visited_urls:
                continue

            if current_url in self.seen_pages:
                continue
            
            visited_urls.add(current_url)
            self.seen_pages.add(current_url)
            pages_processed += 1
            
            limit_label = page_limit
            logger.info(f"Procesando página {pages_processed}/{limit_label}: {current_url}")
            
            try:
                # Descargar página
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Error descargando {current_url}: {e}")
                continue
            
            # Extraer imágenes
            try:
                images_data = self._extract_images_from_html(response.text, current_url)
                logger.info(f"Encontradas {len(images_data)} imágenes en {current_url}")
            except Exception as e:
                logger.error(f"Error extrayendo imágenes de {current_url}: {e}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            for link_url in self._extract_links_from_page(current_url, soup):
                normalized = self._normalize_internal_link(current_url, link_url, allowed_domains)
                if normalized and normalized not in visited_urls and normalized not in to_visit:
                    to_visit.append(normalized)
            to_visit = self._prioritize_to_visit_queue(to_visit, visited_urls)
            
            # Procesar cada imagen
            for img_data in images_data:
                self._process_single_image(img_data)
            
            # Respetar el servidor
            time.sleep(self.delay)
        
        # Generar reportes
        self._generate_reports()
        
        return {
            'total_images_found': len(self.downloaded_images),
            'total_failed': len(self.failed_urls),
            'pages_processed': pages_processed,
            'output_directory': str(self.output_dir)
        }
    
    def _process_single_image(self, image_data: Dict):
        """Procesa una imagen individual"""
        image_url = image_data['image_url']
        
        # Descargar
        download_result = self._download_image(image_data)
        if not download_result:
            return
        
        image_bytes, content_type, response_headers = download_result
        
        # Validar calidad
        is_valid, width, height, format_type = self._validate_image_quality(image_bytes)
        if not is_valid:
            logger.warning(f"Imagen no cumple criterios de calidad: {image_url}")
            return

        # Deduplicación perceptual: detecta equivalentes visuales aunque cambie
        # tamaño, compresión o variante de presentación del CMS.
        perceptual_hash_value = self._compute_dhash(image_bytes)
        if perceptual_hash_value is not None:
            perceptual_duplicate = self._find_perceptual_duplicate(perceptual_hash_value)
            if perceptual_duplicate:
                matched_url, distance = perceptual_duplicate
                self._register_duplicate(
                    image_url,
                    'perceptual_duplicate',
                    matched_with=matched_url,
                    extra={'distance': distance, 'threshold': self.perceptual_threshold}
                )
                logger.debug(f"Imagen duplicada perceptual: {image_url} (distancia={distance})")
                return
        
        # Verificar duplicados por hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash in self.seen_hashes:
            logger.debug(f"Imagen duplicada (mismo hash): {image_url}")
            return
        
        self.seen_hashes.add(image_hash)
        
        # Extraer fecha de la imagen
        image_timestamp, image_date = self._extract_image_date(image_bytes, response_headers)
        
        # Crear metadatos
        metadata = ImageMetadata(
            image_id=hashlib.md5(f"{image_url}{image_date or 'unknown'}".encode()).hexdigest(),
            image_url=image_url,
            source_domain=image_data['source_domain'],
            source_url=image_data['page_url'],
            title=image_data.get('title', ''),
            alt_text=image_data.get('alt_text', ''),
            image_type=content_type,
            filename=image_data.get('image_name', 'image'),
            width=width,
            height=height,
            file_size_kb=image_data.get('file_size_kb', 0.0),
            image_hash=image_hash,
            html_tag=image_data.get('html_tag', ''),
            parent_tag=image_data.get('parent_tag', ''),
            classes=image_data.get('classes', ''),
            element_id=image_data.get('element_id', ''),
            image_timestamp=image_timestamp,
            image_date=image_date,
            image_name_from_url=image_data.get('image_name', ''),
            canonical_image_key=image_data.get('canonical_image_key', ''),
            perceptual_hash=(
                f"{perceptual_hash_value:0{(self.perceptual_hash_size * self.perceptual_hash_size) // 4}x}"
                if perceptual_hash_value is not None
                else ""
            ),
            srcset=image_data.get('srcset', ''),
            loading_strategy=image_data.get('loading', '')
        )
        
        # Guardar imagen
        if self.convert_to_png:
            image_bytes, format_type = self._convert_to_png(image_bytes)
            if not image_bytes:
                logger.warning(f"No se pudo convertir a PNG: {image_url}")
                return

        image_extension = 'png' if self.convert_to_png else format_type.lower()
        image_filename = f"{metadata.image_id}.{image_extension}"
        image_path = self.output_dir / 'images_lavanguardia' / image_filename
        
        try:
            image_path.write_bytes(image_bytes)
            metadata.filename = image_filename
            metadata.output_format = image_extension
            self.downloaded_images.append(metadata)
            if perceptual_hash_value is not None:
                self.seen_perceptual_hashes.append((perceptual_hash_value, image_url))
            logger.info(f"Imagen guardada: {image_filename}")
        except Exception as e:
            logger.error(f"Error guardando imagen {image_url}: {e}")

    def _image_quality_score(self, img: ImageMetadata) -> Tuple[int, float, int, int]:
        """Score de calidad para decidir qué versión conservar."""
        return (
            img.width * img.height,
            img.file_size_kb,
            img.width,
            img.height,
        )

    def _get_context_group_key(self, img: ImageMetadata) -> Optional[str]:
        """Genera clave de agrupación por contexto semántico y recurso similar."""
        source_url = (img.source_url or "").split('#')[0].strip().lower()
        domain = (img.source_domain or "").strip().lower()
        context_text = _normalize_context_text(img.title or img.alt_text)
        asset_signature = _extract_asset_signature(img.image_name_from_url, img.image_url)

        # Prioridad 1: misma firma de recurso (variantes responsive)
        # Permite agrupar versiones con/sin título/alt del mismo asset.
        if asset_signature:
            return f"{domain}|{source_url}|asset:{asset_signature}"

        # Prioridad 2: mismo contexto editorial en la misma página
        if context_text:
            return f"{domain}|{source_url}|ctx:{context_text}"

        return None

    def _deduplicate_by_context(self):
        """Deduplica imágenes por canonical_image_key y conserva la de mejor calidad.

        Esta capa actúa como red de seguridad por si alguna variante se coló en
        el flujo de extracción. No usa hash binario como criterio principal.
        """
        if not self.downloaded_images:
            return

        grouped: Dict[str, List[ImageMetadata]] = defaultdict(list)
        keep_without_group: List[ImageMetadata] = []

        for img in self.downloaded_images:
            key = img.canonical_image_key or canonicalize_image_url(img.image_url)
            if key:
                grouped[key].append(img)
            else:
                keep_without_group.append(img)

        deduped_images: List[ImageMetadata] = keep_without_group.copy()
        removed = 0

        for _, variants in grouped.items():
            if len(variants) == 1:
                deduped_images.append(variants[0])
                continue

            best = max(variants, key=self._image_quality_score)
            deduped_images.append(best)

            for candidate in variants:
                if candidate.image_id == best.image_id:
                    continue
                removed += 1
                image_path = self.output_dir / 'images_lavanguardia' / candidate.filename
                try:
                    if image_path.exists():
                        image_path.unlink()
                except Exception as e:
                    logger.debug(f"No se pudo eliminar duplicado {image_path}: {e}")

        if removed > 0:
            logger.info(
                f"Deduplicación por contexto completada: {removed} variantes descartadas; "
                f"{len(deduped_images)} imágenes finales"
            )

        self.downloaded_images = deduped_images

    def _convert_to_png(self, image_bytes: bytes) -> Tuple[bytes, str]:
        """Convierte cualquier imagen soportada a PNG."""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    if 'A' in img.getbands():
                        img = img.convert('RGBA')
                    else:
                        img = img.convert('RGB')
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                return buffer.getvalue(), 'PNG'
        except Exception as e:
            logger.debug(f"Error convirtiendo a PNG: {e}")
            return b'', ''
    
    def _generate_reports(self):
        """Genera reportes CSV y JSON con los metadatos"""
        self._deduplicate_by_context()
        
        # CSV principal
        csv_path = self.output_dir / 'metadata' / 'images_metadata.csv'
        self._write_csv(csv_path, [img.to_dict() for img in self.downloaded_images])
        logger.info(f"Metadatos guardados en: {csv_path}")
        
        # JSON detallado
        json_path = self.output_dir / 'metadata' / 'images_metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [img.to_dict() for img in self.downloaded_images],
                f,
                ensure_ascii=False,
                indent=2
            )
        logger.info(f"Metadatos JSON guardados en: {json_path}")
        
        # Reporte por dominio
        by_domain = defaultdict(list)
        for img in self.downloaded_images:
            by_domain[img.source_domain].append(img.to_dict())
        
        for domain, images in by_domain.items():
            domain_slug = domain.replace('.', '_')
            domain_csv = self.output_dir / 'metadata' / f'images_{domain_slug}.csv'
            self._write_csv(domain_csv, images)
        
        # Reporte resumen
        duplicate_reasons = sorted(set(event.get('reason', '') for event in self.duplicate_events if event.get('reason')))
        summary = {
            'scraping_timestamp': datetime.now().isoformat(),  # Fecha del scraping
            'total_images': len(self.downloaded_images),
            'total_failed': len(self.failed_urls),
            'total_duplicates_detected': len(self.duplicate_events),
            'duplicates_by_reason': {
                reason: sum(1 for event in self.duplicate_events if event.get('reason') == reason)
                for reason in duplicate_reasons
            },
            'total_unique_domains': len(set(img.source_domain for img in self.downloaded_images)),
            'average_image_size_kb': sum(img.file_size_kb for img in self.downloaded_images) / max(len(self.downloaded_images), 1),
            'image_formats': list(set(img.image_type for img in self.downloaded_images)),
            'domains': list(set(img.source_domain for img in self.downloaded_images)),
            'date_range': {
                'earliest': min((img.image_date for img in self.downloaded_images if img.image_date), default='desconocida'),
                'latest': max((img.image_date for img in self.downloaded_images if img.image_date), default='desconocida')
            }
        }
        
        summary_path = self.output_dir / 'metadata' / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Reporte resumen: {summary}")
        logger.info(f"Archivos guardados en: {self.output_dir}")
    
    def _write_csv(self, path: Path, data: List[Dict]):
        """Escribe datos en formato CSV"""
        if not data:
            return
        
        fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Web Image Scraper - Extrae imágenes con metadatos detallados'
    )
    parser.add_argument('--url', required=True, help='URL del sitio a scrapear')
    parser.add_argument('--output-dir', default='./image_scraper_results', help='Directorio de salida')
    parser.add_argument(
        '--max-pages',
        type=int,
        default=1,
        help=f'Número máximo de páginas a procesar (máximo real: {SAFE_MAX_PAGES}; 0 o negativo usa ese límite)'
    )
    parser.add_argument('--delay', type=float, default=1.0, help='Delay entre peticiones (segundos)')
    parser.add_argument('--crawl-site', action='store_true', help='Intentar descubrir y recorrer el sitio completo, no solo la URL inicial')
    parser.add_argument('--convert-to-png', action='store_true', help='Convertir todas las imágenes descargadas a PNG')
    parser.add_argument(
        '--perceptual-threshold',
        type=int,
        default=DEFAULT_PERCEPTUAL_THRESHOLD,
        help='Umbral de distancia de Hamming para deduplicación perceptual (dHash)'
    )
    
    args = parser.parse_args()
    
    scraper = AdvancedImageScraper(
        output_dir=args.output_dir,
        delay=args.delay,
        convert_to_png=args.convert_to_png,
        perceptual_threshold=args.perceptual_threshold,
    )
    result = scraper.scrape(args.url, max_pages=args.max_pages, crawl_site=args.crawl_site)
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETADO")
    print("="*60)
    print(f"Imágenes descargadas: {result['total_images_found']}")
    print(f"Fallos: {result['total_failed']}")
    print(f"Páginas procesadas: {result['pages_processed']}")
    print(f"Resultados en: {result['output_directory']}")
    print("="*60)


if __name__ == '__main__':
    main()
