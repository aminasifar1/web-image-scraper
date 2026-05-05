#!/usr/bin/env python3
"""
download_wayback_images.py

Descarga las imágenes de los snapshots ya recolectados.

Lee los meta.json y descarga todas las imágenes encontradas.
Las organiza en: wayback_results/domain/year/timestamp/images/

Uso:
    python download_wayback_images.py ./wayback_results

Opciones:
    --max-workers: threads paralelos (default: 8)
    --timeout: timeout por descarga en segundos (default: 30)
    --retry: reintentos por imagen (default: 3)
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger("download_images")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def download_image(url: str, output_path: Path, timeout: int = 30, max_retries: int = 3) -> Tuple[bool, str]:
    """Descarga una imagen y la guarda.
    
    Args:
        url: URL de la imagen
        output_path: dónde guardarla
        timeout: timeout en segundos
        max_retries: número de reintentos
        
    Returns:
        Tupla (éxito, mensaje)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            
            # Guardar imagen
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True, f"✓ Descargada"
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, "Conexión fallida"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, str(type(e).__name__)
    
    return False, "Falló tras reintentos"


def extract_filename_from_url(url: str) -> str:
    """Extrae nombre de archivo desde URL."""
    parsed = urlparse(url)
    filename = parsed.path.split('/')[-1]
    
    # Si no hay extensión clara, usar hash
    if not filename or '.' not in filename:
        import hashlib
        filename = hashlib.md5(url.encode()).hexdigest()[:12] + '.jpg'
    
    return filename


def process_snapshot(snapshot_dir: Path, output_base: Path, 
                    timeout: int = 30, max_workers: int = 4) -> Dict:
    """Procesa un snapshot y descarga sus imágenes.
    
    Args:
        snapshot_dir: directorio del snapshot (ej: wayback_results/domain/2024/20240101/)
        output_base: base para guardar las imágenes
        timeout: timeout para descargas
        max_workers: workers paralelos
        
    Returns:
        Diccionario con estadísticas
    """
    meta_file = snapshot_dir / 'meta.json'
    if not meta_file.exists():
        return {'skipped': True, 'reason': 'Sin meta.json'}
    
    try:
        with meta_file.open('r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        return {'skipped': True, 'reason': f'Error leyendo meta.json: {e}'}
    
    images = meta.get('images', [])
    if not images:
        return {'total': 0, 'downloaded': 0, 'failed': 0}
    
    # Crear directorio para imágenes
    images_dir = snapshot_dir / 'images'
    
    results = {'total': len(images), 'downloaded': 0, 'failed': 0, 'errors': []}
    
    # Descargar en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for url in images:
            filename = extract_filename_from_url(url)
            output_path = images_dir / filename
            
            # Skip si ya existe
            if output_path.exists():
                results['downloaded'] += 1
                continue
            
            future = executor.submit(download_image, url, output_path, timeout)
            futures[future] = (url, output_path, filename)
        
        # Procesar resultados
        for future in as_completed(futures):
            url, output_path, filename = futures[future]
            try:
                success, msg = future.result()
                if success:
                    results['downloaded'] += 1
                    logger.debug(f"  {filename}: {msg}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{filename}: {msg}")
                    logger.warning(f"  {filename}: {msg}")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{filename}: {str(e)}")
                logger.error(f"  {filename}: {str(e)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Descarga imágenes de snapshots de Wayback Machine'
    )
    parser.add_argument('results_dir', default='./wayback_results',
                       help='Directorio con resultados (default: %(default)s)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Threads paralelos (default: %(default)s)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout por descarga en segundos (default: %(default)s)')
    parser.add_argument('--retry', type=int, default=3,
                       help='Reintentos por imagen (default: %(default)s)')
    parser.add_argument('--domain', help='Filtrar por dominio (ej: elpais.com)')
    parser.add_argument('--year', type=int, help='Filtrar por año (ej: 2024)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Output detallado')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Directorio no encontrado: {results_dir}")
        return
    
    logger.info(f"Buscando snapshots en {results_dir}")
    
    # Encontrar todos los snapshots
    snapshots = []
    for domain_dir in results_dir.glob('*'):
        if not domain_dir.is_dir():
            continue
        
        # Filtrar dominio
        if args.domain and args.domain not in domain_dir.name:
            continue
        
        for year_dir in domain_dir.glob('*'):
            if not year_dir.is_dir():
                continue
            
            # Filtrar año
            year = year_dir.name
            if args.year and str(args.year) != year:
                continue
            
            for timestamp_dir in year_dir.glob('*'):
                if not timestamp_dir.is_dir():
                    continue
                snapshots.append(timestamp_dir)
    
    if not snapshots:
        logger.error("No snapshots encontrados")
        return
    
    logger.info(f"Encontrados {len(snapshots)} snapshots")
    
    # Procesar cada snapshot
    global_stats = {
        'total_images': 0,
        'total_downloaded': 0,
        'total_failed': 0,
        'snapshots_processed': 0,
    }
    
    for i, snapshot_dir in enumerate(sorted(snapshots), 1):
        logger.info(f"[{i}/{len(snapshots)}] Procesando {snapshot_dir.relative_to(results_dir)}")
        
        stats = process_snapshot(
            snapshot_dir,
            results_dir,
            timeout=args.timeout,
            max_workers=args.max_workers
        )
        
        if stats.get('skipped'):
            logger.info(f"  ⊘ Saltado: {stats['reason']}")
            continue
        
        global_stats['snapshots_processed'] += 1
        global_stats['total_images'] += stats['total']
        global_stats['total_downloaded'] += stats['downloaded']
        global_stats['total_failed'] += stats['failed']
        
        logger.info(f"  ✓ {stats['downloaded']}/{stats['total']} imágenes "
                   f"(fallos: {stats['failed']})")
        
        if stats.get('errors') and len(stats['errors']) <= 3:
            for error in stats['errors'][:3]:
                logger.debug(f"    - {error}")
    
    # Resumen final
    logger.info("\n" + "="*60)
    logger.info("RESUMEN FINAL")
    logger.info("="*60)
    logger.info(f"Snapshots procesados: {global_stats['snapshots_processed']}")
    logger.info(f"Total imágenes encontradas: {global_stats['total_images']}")
    logger.info(f"Imágenes descargadas: {global_stats['total_downloaded']}")
    logger.info(f"Fallos: {global_stats['total_failed']}")
    
    if global_stats['total_images'] > 0:
        pct = (global_stats['total_downloaded'] / global_stats['total_images']) * 100
        logger.info(f"Tasa de éxito: {pct:.1f}%")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()
