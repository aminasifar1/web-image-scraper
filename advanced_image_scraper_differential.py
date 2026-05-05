#!/usr/bin/env python3
"""
Elimina imágenes duplicadas basándose en el CSV generado por el scraper.
Soporta detección por:
- Dimensiones (ancho x alto) + nombre base de archivo
- Hash MD5 del contenido (exacto)
- Conserva el primer archivo de cada grupo o el más antiguo (--keep-oldest)

Uso:
    python deduplicate_images.py --csv resultados.csv
    python deduplicate_images.py --csv resultados.csv --dry-run
    python deduplicate_images.py --csv resultados.csv --use-md5 --keep-oldest
"""

import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


def get_base_filename_from_url(url: str) -> str:
    """
    Extrae el nombre del archivo desde la URL (ignorando query parameters).
    Ejemplo: 'https://.../projects/808_webp/3ad82d...jpg' -> '3ad82d...jpg'
    """
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path
    filename = Path(path).name
    return filename


def get_image_signature(row: dict, fieldnames: list) -> tuple:
    """
    Genera una firma única para detectar duplicados.
    Por defecto usa (width, height, nombre_base_de_imagen).
    Si no hay width/height, usa solo nombre_base.
    """
    # Buscar columnas de ancho y alto (distintos nombres posibles)
    width = None
    height = None
    for col in fieldnames:
        col_lower = col.lower()
        if col_lower in ('width', 'ancho', 'w'):
            width = row.get(col)
        if col_lower in ('height', 'alto', 'h'):
            height = row.get(col)

    # Obtener URL de imagen desde varias columnas posibles
    url = ""
    for col in fieldnames:
        if col.lower() in ('image_url', 'url', 'img_url', 'source_url'):
            url = row.get(col, "")
            break
    base_name = get_base_filename_from_url(url)

    try:
        # Convertir a int si es posible
        w = int(width) if width and str(width).isdigit() else None
        h = int(height) if height and str(height).isdigit() else None
        if w is not None and h is not None:
            return (w, h, base_name)
        else:
            return (base_name,)
    except (TypeError, ValueError):
        return (base_name,)


def md5_file(filepath: Path) -> str:
    """Calcula MD5 de un archivo."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
    except Exception:
        return ""
    return hash_md5.hexdigest()


def find_path_column(fieldnames: list) -> str:
    """Detecta la columna que contiene la ruta local del archivo."""
    candidates = ['image_path', 'local_path', 'file_path', 'path', 'ruta_local', 'downloaded_path']
    for col in candidates:
        if col in fieldnames:
            return col
    # Si no se encuentra, sugerir la primera que contenga 'path'
    for col in fieldnames:
        if 'path' in col.lower():
            return col
    return ""


def main():
    parser = argparse.ArgumentParser(description="Deduplicar imágenes desde CSV")
    parser.add_argument("--csv", required=True, help="Archivo CSV con los metadatos")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin borrar")
    parser.add_argument("--use-md5", action="store_true", help="Usar hash MD5 para detección exacta")
    parser.add_argument("--keep-oldest", action="store_true", help="Conservar el archivo más antiguo (por modificación)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: No se encuentra el archivo {csv_path}")
        sys.exit(1)

    # Leer CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print("El CSV está vacío.")
        return

    path_col = find_path_column(fieldnames)
    if not path_col:
        print("No se encontró una columna de ruta de archivo en el CSV.")
        print(f"Columnas disponibles: {', '.join(fieldnames)}")
        sys.exit(1)

    print(f"Usando columna de ruta: '{path_col}'")
    print(f"Total de registros en CSV: {len(rows)}")

    # -------- 1. Agrupación por firma (dimensiones + nombre base) --------
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        sig = get_image_signature(row, fieldnames)
        if sig is None or sig == ("",):
            continue
        groups[sig].append((idx, row))

    to_delete = []  # lista de (idx_fila, row, Path)
    kept_count = 0

    print("\n--- Agrupando por firma (ancho x alto + nombre base) ---")
    for sig, items in groups.items():
        if len(items) <= 1:
            kept_count += len(items)
            continue

        print(f"\nFirma duplicada: {sig}")
        # Resolver rutas de archivo
        file_infos = []
        for idx, row in items:
            raw_path = row.get(path_col, "").strip()
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.exists():
                print(f"  [FALTA] {raw_path}")
                continue
            mtime = p.stat().st_mtime if args.keep_oldest else None
            file_infos.append((idx, row, p, mtime))

        if len(file_infos) <= 1:
            kept_count += len(file_infos)
            continue

        # Decidir cuál conservar
        if args.keep_oldest:
            # Ordenar por fecha de modificación (más antigua primero)
            file_infos.sort(key=lambda x: x[3])
            keep = file_infos[0]
            to_del = file_infos[1:]
        else:
            # Conservar el primero según orden original en CSV
            keep = file_infos[0]
            to_del = file_infos[1:]

        keep_idx, keep_row, keep_path, _ = keep
        print(f"  CONSERVAR: {keep_path}")
        for idx, row, p, _ in to_del:
            print(f"  ELIMINAR: {p}")
            to_delete.append((idx, row, p))
        kept_count += 1

    # -------- 2. Opcional: detección exacta por MD5 (entre todos los archivos) --------
    if args.use_md5:
        print("\n--- Verificando duplicados exactos por MD5 ---")
        md5_map = defaultdict(list)
        # Recolectar todos los archivos existentes (que no estén ya marcados para eliminar)
        all_files = []
        for idx, row in enumerate(rows):
            raw_path = row.get(path_col, "").strip()
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.exists():
                continue
            # Evitar volver a procesar si ya está en to_delete
            if any(p == dp for (_, _, dp) in to_delete):
                continue
            all_files.append((idx, row, p))

        for idx, row, p in all_files:
            md5 = md5_file(p)
            if md5:
                md5_map[md5].append((idx, row, p))

        for md5, items in md5_map.items():
            if len(items) <= 1:
                continue
            print(f"\nMD5 duplicado: {md5[:16]}...")
            # Conservar el primero (o el más antiguo si se pide)
            if args.keep_oldest:
                items.sort(key=lambda x: x[2].stat().st_mtime)
                keep = items[0]
                to_del = items[1:]
            else:
                keep = items[0]
                to_del = items[1:]

            keep_idx, keep_row, keep_path = keep
            print(f"  CONSERVAR (MD5): {keep_path}")
            for idx, row, p in to_del:
                # Evitar duplicados en la lista to_delete
                if not any(p == dp for (_, _, dp) in to_delete):
                    to_delete.append((idx, row, p))
                    print(f"  ELIMINAR (MD5): {p}")

    # -------- 3. Ejecutar eliminación o simulación --------
    if args.dry_run:
        print("\n[ SIMULACIÓN ] Archivos que se eliminarían:")
        for idx, row, p in to_delete:
            print(f"  {p}")
        print(f"\nTotal a eliminar: {len(to_delete)}")
        print(f"Archivos que quedarían: {kept_count}")
        return

    # Eliminar realmente
    deleted_files = 0
    for idx, row, p in to_delete:
        try:
            if p.exists():
                p.unlink()
                deleted_files += 1
                print(f"Eliminado: {p}")
            else:
                print(f"No existe: {p}")
        except Exception as e:
            print(f"Error al eliminar {p}: {e}")

    # -------- 4. Actualizar CSV (eliminar filas de archivos borrados) --------
    new_rows = []
    deleted_rows = 0
    # Construir conjunto de rutas eliminadas
    deleted_paths = {str(p) for (_, _, p) in to_delete}

    for row in rows:
        raw_path = row.get(path_col, "").strip()
        if raw_path in deleted_paths:
            deleted_rows += 1
            continue
        new_rows.append(row)

    # Guardar copia de seguridad
    backup_path = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup_path)
    print(f"\nCopia de seguridad guardada en: {backup_path}")

    # Escribir CSV limpio
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"CSV actualizado: {len(new_rows)} filas (se eliminaron {deleted_rows} filas duplicadas)")
    print(f"Archivos eliminados físicamente: {deleted_files}")
    print(f"Archivos únicos conservados: {kept_count}")


if __name__ == "__main__":
    main()