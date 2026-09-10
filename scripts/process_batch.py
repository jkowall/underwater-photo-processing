"""Reusable NEF correction runner. All pixel work is local; no model calls."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import time
from importlib.metadata import version
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw
from underwater_pipeline import (load_source, metadata, preview_size, estimate,
    color_correct, protect_highlights, lab_to_rgb_safe, polish, richer)

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def recipe_id(look):
    root=Path(__file__).parent
    value={'look':look,'code':[sha256(root/name) for name in ['process_batch.py','underwater_pipeline.py']],
           'dependencies':{name:version(name) for name in ['numpy','opencv-python-headless','rawpy','Pillow']}}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def write_json(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)

def resume_valid(path,output,look,recipe):
    dest=output/(path.stem+'_'+look+'.png')
    report=output/'reports'/(path.stem+'.json')
    if not (dest.is_file() and report.is_file()):return False
    try:
        saved=json.loads(report.read_text(encoding='utf-8'))
        return (saved.get('recipe_id')==recipe and saved.get('source_sha256')==sha256(path)
                and saved.get('output_sha256')==sha256(dest)
                and (output/'previews'/(path.stem+'.jpg')).is_file())
    except (OSError,ValueError):return False

def initial(rgb):
    h,w=rgb.shape[:2]
    params=estimate(preview_size(rgb,1500))
    lab=np.empty((h,w,3),np.float32)
    for start in range(0,h,256):
        lab[start:start+256]=cv2.cvtColor(color_correct(rgb[start:start+256],params),cv2.COLOR_RGB2LAB)
    improved=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(12,8)).apply(np.round(lab[:,:,0]*2.55).astype(np.uint8))
    result=np.empty_like(rgb)
    for start in range(0,h,256):
        tile=lab[start:start+256]
        tile[:,:,0]=.5*tile[:,:,0]+.5*improved[start:start+256]/2.55
        result[start:start+256]=np.round(lab_to_rgb_safe(protect_highlights(tile,rgb[start:start+256]))*255).astype(np.uint8)
    return result,params

def stats(rgb):
    means,std=cv2.meanStdDev(rgb)
    result={}
    for i,name in enumerate(('Red','Green','Blue')):
        hist=cv2.calcHist([rgb],[i],None,[256],[0,256]).ravel().astype(np.int64)
        result[name]={'mean_8bit':float(means[i,0]),'std_8bit':float(std[i,0]),
                      'histogram':hist.tolist(),'zero_percent':float(100*hist[0]/(rgb.shape[0]*rgb.shape[1])),
                      'highlight_percent':float(100*hist[255]/(rgb.shape[0]*rgb.shape[1]))}
    return result

def process(path,output,look,recipe):
    started=time.monotonic()
    source_hash=sha256(path)
    original=load_source(path)
    result,params=initial(original)
    cleanup=None
    if look in ('pop','vivid'):
        result,mask,particles,black=polish(result)
        cleanup={'candidates':len(particles),'pixel_percent':100*float(np.count_nonzero(mask))/mask.size,'black_offset':black}
        del mask
    if look=='vivid':
        for start in range(0,len(result),256):result[start:start+256]=richer(result[start:start+256])
    dest=output/(path.stem+'_'+look+'.png')
    exif,date=metadata(path,result.shape[1],result.shape[0]);exif[305]='Python underwater correction: '+look
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tmp=dest.with_suffix('.tmp.png')
    Image.fromarray(result).save(tmp,icc_profile=icc,exif=exif)
    with Image.open(tmp) as saved:
        assert saved.size==original.shape[1::-1]
        assert saved.info.get('icc_profile')==icc
        actual_date=saved.getexif().get_ifd(34665).get(36867,saved.getexif().get(306))
        assert actual_date==date
        saved.verify()
    decoded=cv2.cvtColor(cv2.imread(str(tmp)),cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded,result)
    del decoded
    tmp.replace(dest)
    a=Image.fromarray(preview_size(original,1000));b=Image.fromarray(preview_size(result,1000))
    comparison=Image.new('RGB',(2000,a.height+48),'#171b20')
    comparison.paste(a,(0,48));comparison.paste(b,(1000,48))
    draw=ImageDraw.Draw(comparison)
    draw.text((15,12),path.stem+' | source camera JPEG',fill='white')
    draw.text((1015,12),'Corrected | '+look,fill='white')
    comparison.save(output/'previews'/(path.stem+'.jpg'),quality=94,icc_profile=icc)
    if sha256(path)!=source_hash:raise RuntimeError('Source changed during processing; output cannot be reused')
    report={'source':path.name,'output':dest.name,'look':look,
            'recipe_id':recipe,'source_sha256':source_hash,'output_sha256':sha256(dest),
            'representation':'Full-resolution embedded camera JPEG, assumed sRGB',
            'dimensions':list(result.shape[1::-1]),'capture_date':str(date),'parameters':params,
            'cleanup':cleanup,'before':stats(original),'after':stats(result),
            'seconds':round(time.monotonic()-started,2),
            'validation':'Exact PNG pixel round trip, dimensions, capture date and ICC verified'}
    write_json(output/'reports'/(path.stem+'.json'),report)
    print('Verified',dest.name,flush=True)
    return {'name':dest.name,'dimensions':report['dimensions'],'bytes':dest.stat().st_size}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--look',choices=['natural','pop','vivid'],default='vivid')
    p.add_argument('--resume',action='store_true',help='Skip outputs matching source, recipe, dependencies and PNG checksums')
    args=p.parse_args()
    if not args.input.exists():p.error('Input does not exist')
    paths=[args.input] if args.input.is_file() else sorted(x for x in args.input.iterdir() if x.suffix.lower()=='.nef')
    if not paths or any(x.suffix.lower()!='.nef' for x in paths):p.error('Input must contain NEF files')
    if len({x.stem.casefold() for x in paths})!=len(paths):p.error('Duplicate input stems would collide')
    for directory in [args.output,args.output/'previews',args.output/'reports']:directory.mkdir(parents=True,exist_ok=True)
    lock=args.output/'.processing.lock'
    try:lock.mkdir()
    except FileExistsError:p.error('Output is locked. If the previous run was killed, confirm it stopped before removing .processing.lock')
    try:
        (lock/'pid').write_text(str(os.getpid()),encoding='utf-8')
        recipe=recipe_id(args.look)
        results=[];errors=[];skip=set()
        for path in paths:
            occupied=any(x.exists() for x in [args.output/(path.stem+'_'+args.look+'.png'),args.output/'reports'/(path.stem+'.json'),args.output/'previews'/(path.stem+'.jpg')])
            if occupied:
                if args.resume and resume_valid(path,args.output,args.look,recipe):skip.add(path)
                else:p.error(f'Unverified output collision for {path.name}; choose a fresh output folder')
        for path in paths:
            try:
                if path in skip:
                    name=path.stem+'_'+args.look+'.png'
                    results.append({'name':name,'status':'verified-existing'})
                    print('Verified existing',name,flush=True)
                else:results.append(process(path,args.output,args.look,recipe))
            except Exception as error:
                errors.append({'source':path.name,'error':str(error)})
                print('Failed',path.name,type(error).__name__,str(error),flush=True)
            write_json(args.output/'batch_summary.json',{'look':args.look,'recipe_id':recipe,'completed':results,'errors':errors})
    finally:
        (lock/'pid').unlink(missing_ok=True)
        lock.rmdir()
    print(f'Completed {len(results)}/{len(paths)}',flush=True)
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
