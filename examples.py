#!/usr/bin/env python3
"""
Ejemplos de uso del Advanced Image Scraper
Demuestra diferentes formas de usar el scraper para diferentes casos de uso
"""

import json
import csv
from pathlib import Path
from advanced_image_scraper import AdvancedImageScraper


def example_1_basic_scraping():
    """Ejemplo 1: Scraping básico de un sitio web"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Scraping Básico")
    print("="*60)
    
    scraper = AdvancedImageScraper(
        output_dir='./example_results/basic',
        delay=1.0
    )
    
    # Ejemplo con un sitio real (ajusta la URL según necesites)
    result = scraper.scrape(
        url='https://www.wikipedia.org',
        max_pages=1
    )
    
    print(f"\nResultado: {result}")
    print(f"Imágenes descargadas: {result['total_images_found']}")
    print(f"Resultados guardados en: {result['output_directory']}")


def example_2_analyze_metadata():
    """Ejemplo 2: Analizar metadatos descargados"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Análisis de Metadatos")
    print("="*60)
    
    metadata_csv = Path('./example_results/basic/metadata/images_metadata.csv')
    
    if not metadata_csv.exists():
        print("Ejecuta primero el ejemplo 1 para generar metadatos")
        return
    
    # Leer CSV
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        images = list(reader)
    
    print(f"\nTotal de imágenes: {len(images)}")
    
    # Estadísticas
    total_size = sum(float(img.get('file_size_kb', 0)) for img in images)
    avg_size = total_size / len(images) if images else 0
    
    print(f"Tamaño total: {total_size:.2f} KB")
    print(f"Tamaño promedio: {avg_size:.2f} KB")
    
    # Imágenes sin ALT text (accesibilidad)
    missing_alt = [img for img in images if not img.get('alt_text', '')]
    print(f"\nImágenes sin ALT text: {len(missing_alt)}")
    
    # Tipos de imagen
    image_types = {}
    for img in images:
        img_type = img.get('image_type', 'unknown')
        image_types[img_type] = image_types.get(img_type, 0) + 1
    
    print("\nDistribución de tipos:")
    for img_type, count in sorted(image_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {img_type}: {count}")
    
    # Dominios encontrados
    domains = set(img.get('source_domain', '') for img in images)
    print(f"\nDominios únicos: {len(domains)}")
    for domain in sorted(domains):
        count = sum(1 for img in images if img.get('source_domain') == domain)
        print(f"  {domain}: {count} imágenes")


def example_3_export_filtered_metadata():
    """Ejemplo 3: Exportar metadatos filtrados"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Exportar Metadatos Filtrados")
    print("="*60)
    
    metadata_json = Path('./example_results/basic/metadata/images_metadata.json')
    
    if not metadata_json.exists():
        print("Ejecuta primero el ejemplo 1 para generar metadatos")
        return
    
    # Leer JSON
    with open(metadata_json, 'r', encoding='utf-8') as f:
        images = json.load(f)
    
    # Filtro 1: Imágenes grandes (>500 KB)
    large_images = [img for img in images if img['file_size_kb'] > 500]
    print(f"\nImágenes grandes (>500 KB): {len(large_images)}")
    for img in large_images[:3]:
        print(f"  - {img['filename']}: {img['file_size_kb']:.2f} KB")
    
    # Filtro 2: Imágenes en alta resolución (>1920px ancho)
    hires_images = [img for img in images if img['width'] > 1920]
    print(f"\nImágenes en alta resolución (>1920px): {len(hires_images)}")
    
    # Filtro 3: Imágenes en picture tags (responsive)
    responsive = [img for img in images if img['parent_tag'] == 'picture']
    print(f"\nImágenes responsive (picture tag): {len(responsive)}")
    
    # Filtro 4: Imágenes con lazy loading
    lazy_images = [img for img in images if img['loading_strategy'] == 'lazy']
    print(f"\nImágenes con lazy loading: {len(lazy_images)}")
    
    # Exportar filtrados
    output_path = Path('./example_results/basic/metadata/filtered_hires_images.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hires_images, f, ensure_ascii=False, indent=2)
    
    print(f"\nMetadatos filtrados guardados en: {output_path}")


def example_4_quality_analysis():
    """Ejemplo 4: Análisis de calidad de imágenes"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Análisis de Calidad")
    print("="*60)
    
    metadata_json = Path('./example_results/basic/metadata/images_metadata.json')
    
    if not metadata_json.exists():
        print("Ejecuta primero el ejemplo 1 para generar metadatos")
        return
    
    with open(metadata_json, 'r', encoding='utf-8') as f:
        images = json.load(f)
    
    # Análisis de relación de aspecto
    aspects = {}
    for img in images:
        if img['height'] > 0:
            aspect = img['width'] / img['height']
            aspect_cat = "ultrawide"
            if aspect < 0.5:
                aspect_cat = "vertical"
            elif aspect < 1:
                aspect_cat = "portrait"
            elif aspect < 1.5:
                aspect_cat = "cuadrada"
            elif aspect < 2:
                aspect_cat = "landscape"
            elif aspect < 3:
                aspect_cat = "wide"
            
            aspects[aspect_cat] = aspects.get(aspect_cat, 0) + 1
    
    print("\nDistribución de relación de aspecto:")
    for cat, count in sorted(aspects.items()):
        print(f"  {cat}: {count}")
    
    # Imágenes pequeñas (potencial problema)
    small_images = [img for img in images if img['width'] < 200 or img['height'] < 200]
    print(f"\nImágenes pequeñas (<200px): {len(small_images)}")
    
    # Formato PNG vs JPG
    pngs = [img for img in images if 'png' in img['image_type'].lower()]
    jpgs = [img for img in images if 'jpeg' in img['image_type'].lower() or 'jpg' in img['image_type'].lower()]
    
    print(f"\nFormatos:")
    print(f"  PNG: {len(pngs)}")
    print(f"  JPG: {len(jpgs)}")


def example_5_accessibility_report():
    """Ejemplo 5: Reporte de accesibilidad"""
    print("\n" + "="*60)
    print("EJEMPLO 5: Reporte de Accesibilidad")
    print("="*60)
    
    metadata_csv = Path('./example_results/basic/metadata/images_metadata.csv')
    
    if not metadata_csv.exists():
        print("Ejecuta primero el ejemplo 1 para generar metadatos")
        return
    
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        images = list(reader)
    
    # Análisis de ALT text
    no_alt = [img for img in images if not img.get('alt_text', '').strip()]
    short_alt = [img for img in images if 0 < len(img.get('alt_text', '')) < 5]
    good_alt = [img for img in images if len(img.get('alt_text', '')) >= 10]
    
    print("\nANÁLISIS DE ALT TEXT:")
    print(f"  Sin ALT: {len(no_alt)} ({100*len(no_alt)/len(images):.1f}%)")
    print(f"  ALT muy corto (<5 caracteres): {len(short_alt)} ({100*len(short_alt)/len(images):.1f}%)")
    print(f"  ALT descriptivo (>10 caracteres): {len(good_alt)} ({100*len(good_alt)/len(images):.1f}%)")
    
    # Análisis de título
    no_title = [img for img in images if not img.get('title', '').strip()]
    print(f"\nSin atributo 'title': {len(no_title)} ({100*len(no_title)/len(images):.1f}%)")
    
    # Exportar reporte
    report_path = Path('./example_results/basic/metadata/accessibility_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ACCESIBILIDAD\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total de imágenes: {len(images)}\n")
        f.write(f"Sin ALT text: {len(no_alt)} ({100*len(no_alt)/len(images):.1f}%)\n")
        f.write(f"Sin título: {len(no_title)} ({100*len(no_title)/len(images):.1f}%)\n\n")
        f.write("Imágenes sin ALT:\n")
        for img in no_alt[:10]:
            f.write(f"  {img['image_url']}\n")
    
    print(f"\nReporte guardado en: {report_path}")


def example_6_scrape_with_multiple_pages():
    """Ejemplo 6: Scrapear múltiples páginas (requiere más tiempo)"""
    print("\n" + "="*60)
    print("EJEMPLO 6: Scrapear Múltiples Páginas")
    print("="*60)
    print("\nEste ejemplo descargará imágenes de múltiples páginas.")
    print("Puede tomar varios minutos. ¿Deseas continuar? (requiere red)")
    
    # Descomentar para ejecutar
    # scraper = AdvancedImageScraper(output_dir='./example_results/multipage', delay=2.0)
    # result = scraper.scrape(url='https://www.wikipedia.org', max_pages=5)
    # print(f"\nImágenes descargadas de {result['pages_processed']} páginas")


def main():
    """Ejecuta los ejemplos"""
    print("\n" + "="*70)
    print("EJEMPLOS DE USO - Advanced Image Scraper")
    print("="*70)
    
    examples = [
        ("1", "Scraping Básico", example_1_basic_scraping),
        ("2", "Análisis de Metadatos", example_2_analyze_metadata),
        ("3", "Exportar Metadatos Filtrados", example_3_export_filtered_metadata),
        ("4", "Análisis de Calidad", example_4_quality_analysis),
        ("5", "Reporte de Accesibilidad", example_5_accessibility_report),
        ("6", "Múltiples Páginas (Manual)", example_6_scrape_with_multiple_pages),
    ]
    
    print("\nEjemplos disponibles:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    
    print("\n" + "-"*70)
    print("NOTA: Este archivo contiene ejemplos educativos.")
    print("Descomenta las funciones que desees ejecutar en la sección main().")
    print("-"*70)
    
    # Ejecutar ejemplo 2 automáticamente si existen metadatos
    if Path('./example_results/basic/metadata/images_metadata.csv').exists():
        print("\n✓ Se encontraron metadatos previos. Mostrando análisis...\n")
        example_2_analyze_metadata()
    else:
        print("\nPara empezar, ejecuta:")
        print("  python examples.py")
        print("\nLuego descomenta 'example_1_basic_scraping()' en la sección main()")


if __name__ == '__main__':
    main()
    
    # Descomenta los ejemplos que desees ejecutar:
    # example_1_basic_scraping()
    # example_2_analyze_metadata()
    # example_3_export_filtered_metadata()
    # example_4_quality_analysis()
    # example_5_accessibility_report()
    # example_6_scrape_with_multiple_pages()
