#!/usr/bin/env python3
"""Prueba ligera para evaluar sitios de arte.

Lee `websites-list.csv`, hace peticiones HTTP simples y cuenta etiquetas <img>
y URLs de imágenes encontradas. Guarda resultados en `probe_results.csv`.

Uso:
  python quick_art_probe.py --limit 50 --output probe_results.csv
"""
import argparse
import csv
import sys
import time
from urllib.parse import urljoin, urlparse
import ssl

DEFAULT_TIMEOUT = 8

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False
    import urllib.request as _urllib


def read_websites(path):
    urls = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get('url') or '').strip()
            if url:
                urls.append(url)
    return urls


def probe_with_requests(url):
    result = {
        'url': url,
        'status': 'error',
        'http_status': '',
        'n_img_tags': 0,
        'n_unique_img_urls': 0,
        'sample_img_urls': '',
        'elapsed': 0.0,
        'error': ''
    }
    t0 = time.time()
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT, headers={'User-Agent':'Mozilla/5.0'})
        result['http_status'] = int(resp.status_code)
        if resp.status_code != 200 or not resp.text:
            result['status'] = 'bad_status'
            result['elapsed'] = time.time() - t0
            return result

        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')
        img_tags = soup.find_all('img')
        result['n_img_tags'] = len(img_tags)
        img_urls = set()
        samples = []
        for img in img_tags:
            src = img.get('data-src') or img.get('data-original') or img.get('src') or ''
            if not src:
                continue
            abs_url = urljoin(resp.url, src)
            img_urls.add(abs_url)
            if len(samples) < 3:
                samples.append(abs_url)

        result['n_unique_img_urls'] = len(img_urls)
        result['sample_img_urls'] = ' | '.join(samples)
        result['status'] = 'ok'
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
    result['elapsed'] = time.time() - t0
    return result


def probe_fallback(url):
    # Fallback simple: fetch raw and count '<img'
    result = {
        'url': url,
        'status': 'error',
        'http_status': '',
        'n_img_tags': 0,
        'n_unique_img_urls': 0,
        'sample_img_urls': '',
        'elapsed': 0.0,
        'error': ''
    }
    t0 = time.time()
    try:
        req = _urllib.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        # Crear contexto SSL que no verifique el certificado local (útil en macOS sin CA configurada)
        ctx = ssl.create_default_context()
        try:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        except Exception:
            pass
        with _urllib.urlopen(req, timeout=DEFAULT_TIMEOUT, context=ctx) as r:
            data = r.read().decode('utf-8', errors='ignore')
            result['http_status'] = r.getcode()
            count = data.lower().count('<img')
            result['n_img_tags'] = count
            result['status'] = 'ok'
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
    result['elapsed'] = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sites', default='websites-list.csv')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', default='probe_results.csv')
    args = parser.parse_args()

    urls = read_websites(args.sites)
    if not urls:
        print('No se encontraron URLs en', args.sites)
        sys.exit(1)

    to_run = urls if args.limit <= 0 else urls[:args.limit]

    with open(args.output, 'w', newline='', encoding='utf-8') as out_f:
        fieldnames = ['url','status','http_status','n_img_tags','n_unique_img_urls','sample_img_urls','elapsed','error']
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for i, url in enumerate(to_run, 1):
            print(f'[{i}/{len(to_run)}] probing {url} ...', flush=True)
            if HAS_BS4:
                res = probe_with_requests(url)
            else:
                res = probe_fallback(url)
            writer.writerow(res)
            time.sleep(0.6)

    print('Probe completado. Resultados en', args.output)


if __name__ == '__main__':
    main()
