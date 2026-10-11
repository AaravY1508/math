#!/usr/bin/env python3
"""Check exact coordinate round trips, certificate bindings, and artifact integrity."""
import argparse,csv,hashlib,json,pathlib,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def check(root):
    rows=json.loads((root/'summary.json').read_text())['results']
    assert [r['n'] for r in rows]==list(range(36,51)), 'Missing summary cases'
    for row in rows:
        n=row['n'];d=root/f'n{n}'
        s=json.loads((d/'solution.json').read_text());v=json.loads((d/'validation.json').read_text())
        p=np.asarray(s['points'],dtype='<f8')
        assert p.shape==(n,2) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        with (d/'coordinates.csv').open() as f:c=list(csv.DictReader(f))
        pc=np.asarray([[float(r['x']),float(r['y'])] for r in c],dtype='<f8')
        ph=np.asarray([[float.fromhex(x),float.fromhex(y)] for x,y in s['points_hex']],dtype='<f8')
        assert p.tobytes()==pc.tobytes()==ph.tobytes(),f'n{n} round trip mismatch'
        assert all(float(r['brightness'])==1 for r in c)
        h=hashlib.sha256(p.tobytes()).hexdigest()
        assert h==s['coordinates_sha256_float64_little_endian']==v['coordinates_sha256_float64_little_endian']
        assert v['n']==n and v['valid'] and v['certified'] and v['gap']<=1e-7
        assert v['objective_lower']<=v['objective_upper']
        assert v['objective_lower']==row['our_best']==s['objective_lower']
        assert v['objective_upper']==row['objective_upper']==s['objective_upper']
        assert not v.get('oracle_missed_lower_minimum',False)
        assert s['objective_lower']-1e-8<=s['objective_numerical']<=s['objective_upper']+1e-8
        for item in json.loads((d/'manifest.json').read_text())['files']:
            f=d/item['file'];assert f.stat().st_size==item['bytes']
            assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],str(f)
        with Image.open(d/'diagram.png') as im:
            png_size=im.size
            assert png_size==(3500,3727);im.verify()
        with Image.open(d/'diagram.gif') as im:
            assert im.format=='GIF',f'n{n} diagram.gif is not GIF format'
            assert im.n_frames==1,f'n{n} GIF must contain one static frame'
            assert im.size==png_size,f'n{n} GIF and PNG dimensions differ'
            im.load()
        assert ET.parse(d/'diagram.svg').getroot().tag.endswith('svg')
        print(f'n={n}: geometry, decimal/hex coordinates, certificate, hashes, SVG, PNG, GIF PASS')
    print('ALL 15 PACKAGES PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('results',type=pathlib.Path,nargs='?',default=pathlib.Path(__file__).resolve().parent/'results')
    check(ap.parse_args().results)
