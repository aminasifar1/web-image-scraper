#!/usr/bin/env python3
"""Analyze image quality loss from normalization (without actual processing)."""

import csv
from pathlib import Path


def main():
    print("=" * 70)
    print("ANÁLISIS: IMPACTO DE NORMALIZAR IMÁGENES A 224×224 PARA SPAI")
    print("=" * 70)
    print()
    
    # Load original metadata with dimensions
    csv_file = Path("output/quality_filtered/metadata/contextual_images_filtered.csv")
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
    
    # Analyze upscaling requirements
    print(f"IMÁGENES ANALIZADAS: {len(rows)}\n")
    
    # Group by upscaling factor
    groups = {
        "downscale_only": [],  # Downscale only (good)
        "moderate_upscale": [],  # Up to 1.5x (acceptable)
        "significant_upscale": [],  # 1.5-2.5x (some quality loss)
        "heavy_upscale": [],  # >2.5x (significant loss)
    }
    
    quality_scores = []
    
    for row in rows:
        w = int(row.get("detected_width", 0))
        h = int(row.get("detected_height", 0))
        
        if not w or not h:
            continue
        
        # For square fit in 224x224, use the smaller dimension
        smaller_dim = min(w, h)
        upscale_ratio = 224 / smaller_dim
        
        # Quality score (0-1, higher better)
        quality_score = min(1.0 / upscale_ratio, 1.0)
        quality_scores.append(quality_score)
        
        img_info = {
            "name": Path(row.get("filtered_path", "")).name,
            "size": f"{w}x{h}",
            "mp": round(w * h / 1e6, 3),
            "upscale_ratio": round(upscale_ratio, 2),
            "quality_score": round(quality_score, 2),
            "category": row.get("category", "?"),
            "original_dimensions": (w, h),
        }
        
        if upscale_ratio <= 1.0:
            groups["downscale_only"].append(img_info)
        elif upscale_ratio <= 1.5:
            groups["moderate_upscale"].append(img_info)
        elif upscale_ratio <= 2.5:
            groups["significant_upscale"].append(img_info)
        else:
            groups["heavy_upscale"].append(img_info)
    
    # Print grouped analysis
    print("CATEGORÍAS DE IMPACTO AL NORMALIZAR A 224×224:\n")
    
    labels = {
        "downscale_only": "✅ DOWNSCALE ONLY (No quality loss, mejor)",
        "moderate_upscale": "🟢 MODERATE UPSCALE (1.0-1.5x, aceptable)",
        "significant_upscale": "🟡 SIGNIFICANT UPSCALE (1.5-2.5x, pérdida moderada)",
        "heavy_upscale": "🔴 HEAVY UPSCALE (>2.5x, pérdida significativa)",
    }
    
    for key, label in labels.items():
        count = len(groups[key])
        pct = 100 * count / len(rows) if rows else 0
        print(f"{label}")
        print(f"   {count} imágenes ({pct:.1f}%)")
        
        if groups[key]:
            # Show some examples
            examples = sorted(groups[key], key=lambda x: x["upscale_ratio"])[:3]
            for ex in examples:
                print(f"     • {ex['size']:15} → 224×224 (ratio: {ex['upscale_ratio']}x, score: {ex['quality_score']})")
        print()
    
    # Summary statistics
    print("=" * 70)
    print("ESTADÍSTICAS GLOBALES:\n")
    
    avg_ratio = sum(r["upscale_ratio"] for r in sum(groups.values(), []) if "upscale_ratio" in r) / len(rows) if rows else 1
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
    
    print(f"Promedio de upscale ratio: {avg_ratio:.2f}x")
    print(f"Promedio de quality score: {avg_quality:.2f}/1.0")
    print()
    
    # Recommendation
    print("=" * 70)
    print("RECOMENDACIÓN:\n")
    
    heavy_count = len(groups["heavy_upscale"])
    if heavy_count == 0:
        print("✅ TODAS LAS IMÁGENES SON ACEPTABLES para SPAI sin pre-procesamiento")
        print("   La mayoría serán downscaled o moderadamente upscaled → mínima pérdida")
        print("   RECOMENDACIÓN: Usar directamente sin normalizar")
    elif heavy_count <= len(rows) * 0.1:
        print(f"✅ CASI TODAS LAS IMÁGENES SON ACEPTABLES ({heavy_count} problemáticas)")
        print("   RECOMENDACIÓN: Considerar filtrar solo las {heavy_count} muy pequeñas")
    else:
        print(f"⚠️  ALGUNAS IMÁGENES REQUIEREN UPSCALING ({heavy_count} muy pequeñas)")
        print("   RECOMENDACIÓN: Pre-procesamiento opcional para mejor calidad")
    
    print()
    print("=" * 70)
    
    # Distribution by category
    print("\nDISTRIBUCIÓN POR CATEGORÍA:\n")
    
    categories = {}
    for row in rows:
        cat = row.get("category", "?")
        w = int(row.get("detected_width", 0))
        h = int(row.get("detected_height", 0))
        
        smaller_dim = min(w, h) if w and h else 0
        ratio = 224 / smaller_dim if smaller_dim else 1
        
        if cat not in categories:
            categories[cat] = {"count": 0, "ratios": []}
        
        categories[cat]["count"] += 1
        categories[cat]["ratios"].append(ratio)
    
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        avg_cat_ratio = sum(stats["ratios"]) / len(stats["ratios"])
        avg_cat_quality = sum(min(1/r, 1) for r in stats["ratios"]) / len(stats["ratios"])
        
        print(f"  {cat}:")
        print(f"    {stats['count']} imágenes")
        print(f"    Avg upscale ratio: {avg_cat_ratio:.2f}x")
        print(f"    Avg quality score: {avg_cat_quality:.2f}/1.0")
        print()


if __name__ == "__main__":
    main()
