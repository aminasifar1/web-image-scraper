#!/usr/bin/env python3
"""
Script para ejecutar el scraper en todas las URLs del CSV
y guardar resultados en carpetas organizadas por nombre de sitio web.

Uso:
    python batch_scraper.py --csv websites-list.csv
    python batch_scraper.py --csv websites-list.csv --exclude "La Vanguardia"
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def get_domain_name(url: str) -> str:
    """Extrae el nombre del dominio de una URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    return domain.split('/')[0]


def get_safe_folder_name(organization_name: str) -> str:
    """Convierte el nombre de la organización a un nombre de carpeta seguro."""
    # Remover caracteres especiales, espacios y acentos
    safe_name = organization_name.lower()
    safe_name = safe_name.replace(' ', '_')
    safe_name = ''.join(c if c.isalnum() or c == '_' else '' for c in safe_name)
    return safe_name


def main():
    parser = argparse.ArgumentParser(description='Ejecuta scraper para múltiples URLs')
    parser.add_argument('--csv', required=True, help='Archivo CSV con las URLs')
    parser.add_argument('--exclude', default='La Vanguardia', 
                        help='Nombre de organización a excluir')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar URLs sin ejecutar')
    parser.add_argument('--cookies-file', help='Ruta a JSON de cookies de sesión')
    parser.add_argument('--cookie', action='append', default=[], help="Cookie manual name=value (repetible)")
    parser.add_argument('--cookie-header', help='Header completo Cookie del navegador (name=value; name2=value2)')
    parser.add_argument('--browser-cookies', choices=['chrome', 'chromium', 'brave', 'firefox', 'edge', 'safari'], help='Cargar cookies automáticamente desde navegador')
    parser.add_argument('--use-playwright-fallback', action='store_true', help='Usar Playwright si requests falla (403/anti-bot)')
    parser.add_argument('--playwright-headed', action='store_true', help='Abrir navegador visible en fallback Playwright')
    parser.add_argument('--playwright-timeout-ms', type=int, default=30000, help='Timeout de Playwright en milisegundos')
    parser.add_argument('--playwright-wait-until', choices=['load', 'domcontentloaded', 'networkidle', 'commit'], default='networkidle', help='Estrategia de espera de Playwright al navegar')
    parser.add_argument('--header', action='append', default=[], help="Header HTTP 'Clave: Valor' (repetible)")
    parser.add_argument('--images-subdir', default='images', help='Nombre de subcarpeta para imágenes')
    
    args = parser.parse_args()
    
    # Verificar que el CSV existe
    if not os.path.exists(args.csv):
        print(f"Error: No se encontró {args.csv}")
        sys.exit(1)
    
    # Crear carpeta base para resultados
    results_base = Path('scraper_results')
    results_base.mkdir(exist_ok=True)
    
    # Leer y procesar CSV
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            url = row.get('url', '').strip()
            organization_name = row.get('organization_name', '').strip()
            
            # Saltar si no hay URL
            if not url:
                continue
            
            # Saltar URLs excluidas
            if args.exclude in organization_name:
                print(f"⏭️  Saltando: {organization_name} ({url})")
                continue
            
            # Crear carpeta para esta organización
            folder_name = get_safe_folder_name(organization_name)
            output_dir = results_base / folder_name
            output_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"\n{'='*70}")
            print(f"🌐 Procesando: {organization_name}")
            print(f"📍 URL: {url}")
            print(f"📁 Carpeta: {output_dir}")
            print(f"{'='*70}")
            
            if args.dry_run:
                print("[DRY RUN] No se ejecutó nada")
                continue
            
            # Construir comando de scraping
            cmd = [
                'python', 'advanced_image_scraper.py',
                '--url', url,
                '--output-dir', str(output_dir),
                '--images-subdir', args.images_subdir,
            ]

            if args.cookies_file:
                cmd.extend(['--cookies-file', args.cookies_file])

            if args.cookie_header:
                cmd.extend(['--cookie-header', args.cookie_header])

            if args.browser_cookies:
                cmd.extend(['--browser-cookies', args.browser_cookies])

            if args.use_playwright_fallback:
                cmd.append('--use-playwright-fallback')

            if args.playwright_headed:
                cmd.append('--playwright-headed')

            if args.playwright_timeout_ms:
                cmd.extend(['--playwright-timeout-ms', str(args.playwright_timeout_ms)])

            if args.playwright_wait_until:
                cmd.extend(['--playwright-wait-until', args.playwright_wait_until])

            for cookie_pair in args.cookie:
                cmd.extend(['--cookie', cookie_pair])

            for header in args.header:
                cmd.extend(['--header', header])
            
            # Ejecutar scraper
            try:
                result = subprocess.run(cmd, capture_output=False, text=True)
                if result.returncode == 0:
                    print(f"✅ Completado: {organization_name}")
                else:
                    print(f"❌ Error en: {organization_name}")
            except Exception as e:
                print(f"❌ Excepción en {organization_name}: {e}")
    
    print(f"\n{'='*70}")
    print("✨ Procesamiento completado")
    print(f"📁 Resultados guardados en: {results_base}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
