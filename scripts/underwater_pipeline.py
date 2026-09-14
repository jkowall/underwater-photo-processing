"""Reusable pixel-processing functions extracted from the tested local pipeline."""
import cv2
import numpy as np
import rawpy
from PIL import Image, TiffImagePlugin


def linearize(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)

def encode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)

def estimate(rgb):
    x = linearize(rgb.astype(np.float32)/255)
    h,w = x.shape[:2]
    # Low-texture upper corners provide a conservative ambient estimate.
    roi = np.zeros((h,w), bool)
    roi[:int(.28*h),:int(.20*w)] = True
    roi[:int(.28*h),int(.82*w):] = True
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    if not np.any(roi):
        roi[:]=True
    smooth=roi & (texture < np.percentile(texture[roi], 55))
    if np.any(smooth):
        roi=smooth
    ambient = np.percentile(x[roi], 75, axis=0)
    floor = np.percentile(x.reshape(-1,3), 5, axis=0)
    # Ambient and transmission cannot be uniquely inferred here. Bound the
    # additive veil by a fraction of the dark tail, avoiding crushed shadows.
    veil = np.minimum(.07*ambient, .38*floor).astype(np.float32)
    base = x*x/(x+veil+1e-8)
    red_strength=.28
    r0 = restore_red(base, red_strength)
    lum = r0 @ np.array([.2126,.7152,.0722],np.float32)
    valid = (lum>np.percentile(lum,10)) & (lum<np.percentile(lum,98))
    if not np.any(valid):
        valid=np.ones(lum.shape,dtype=bool)
    # Shades of Gray, p=4, calculated in linear RGB on trimmed pixels.
    norms = np.mean(r0[valid]**4,axis=0)**.25
    gains = np.mean(norms)/np.maximum(norms,1e-5) if np.max(norms)>1e-5 else np.ones(3)
    gains = np.clip(gains,.4,12).astype(np.float32)
    return {'ambient_linear_RGB':ambient.tolist(), 'veil_linear_RGB':veil.tolist(),
            'shades_of_gray_p':4, 'white_balance_gains_RGB':gains.tolist(),
            'red_compensation_strength':red_strength}

def restore_red(x, strength=.28):
    # Regularized missing-red estimate from surviving green structure. It is
    # strongest in midtones, fades near highlights, and preserves measured red.
    y=x.copy()
    peak=np.max(x,axis=2)
    y[:,:,0] += strength*np.maximum(x[:,:,1]-x[:,:,0],0)*(1-.70*np.clip(peak,0,1))
    return y

def color_correct(rgb,params):
    x=linearize(rgb.astype(np.float32)/255)
    veil=np.array(params['veil_linear_RGB'],np.float32)
    # Soft subtraction prevents new hard black clipping.
    x=x*x/(x+veil+1e-8)
    x=restore_red(x, float(params.get('red_compensation_strength',.28)))
    x*=np.array(params['white_balance_gains_RGB'],np.float32)
    # A common scale factor gives highlight rolloff without channel clipping.
    peak=x.max(axis=2)
    shoulder=np.where(peak>.72,.72+.28*(1-np.exp(-(peak-.72)/.28)),peak)
    x*= (shoulder/np.maximum(peak,1e-8))[:,:,None]
    return np.clip(encode(x),0,1)

def lab_to_rgb_safe(lab):
    # Explicit LAB -> XYZ -> linear sRGB avoids OpenCV clipping out-of-gamut
    # channels before we can roll them off after the CLAHE luminance increase.
    fy=(lab[:,:,0]+16)/116
    f=np.stack((fy+lab[:,:,1]/500,fy,fy-lab[:,:,2]/200),axis=2)
    xyz=np.where(f>6/29,f**3,3*(6/29)**2*(f-4/29))
    xyz*=np.array([.95047,1,1.08883],np.float32)
    matrix=np.array([[3.2404542,-1.5371385,-.4985314],
                     [-.9692660,1.8760108,.0415560],
                     [.0556434,-.2040259,1.0572252]],np.float32)
    linear=xyz@matrix.T
    linear=np.maximum(linear,0)
    peak=linear.max(axis=2)
    rolled=np.where(peak>.80,.80+.18*(1-np.exp(-np.maximum(peak-.80,0)/.18)),peak)
    linear*=(rolled/np.maximum(peak,1e-8))[:,:,None]
    return np.clip(encode(linear),0,1)

def load_source(path):
    with rawpy.imread(str(path)) as raw:
        if raw.sizes.flip != 0:
            raise ValueError('Rotated RAW orientation is not supported by this embedded-JPEG recipe')
        thumb=raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            raise ValueError('Expected full-resolution embedded JPEG')
        decoded=cv2.imdecode(np.frombuffer(thumb.data,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError('Embedded JPEG could not be decoded')
        rgb=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
        crop=(raw.sizes.crop_width,raw.sizes.crop_height)
        if rgb.shape[1::-1] != crop:
            raise ValueError(f'Embedded image {rgb.shape[1::-1]} differs from RAW crop {crop}')
    return rgb

def metadata(path,w,h):
    # Copy selected photographic metadata, excluding offsets and maker notes.
    with path.open('rb') as f:
        header=f.read(8)
        ifd=TiffImagePlugin.ImageFileDirectory_v2(header)
        f.seek(ifd.next); ifd.load(f)
        exif=Image.Exif()
        for key in (271,272,306,315,33432):
            if key in ifd: exif[key]=ifd[key]
        sub={}
        if 34665 in ifd:
            f.seek(ifd[34665]); details=TiffImagePlugin.ImageFileDirectory_v2(header); details.load(f)
            for key in (33434,33437,34850,34855,36867,36868,36880,36881,36882,
                        37377,37378,37380,37383,37385,37386,37520,37521,37522,42036):
                if key in details: sub[key]=details[key]
        sub.update({40961:1,40962:w,40963:h})
        exif[34665]=sub
        exif[274]=1
        exif[305]='Python OpenCV underwater correction, first pass'
        return exif,sub.get(36867,ifd.get(306))

def preview_size(rgb,width=1200):
    return cv2.resize(rgb,(width,round(rgb.shape[0]*width/rgb.shape[1])),interpolation=cv2.INTER_AREA)

def classify_look(rgb):
    # Small-preview topside vs underwater guess. Underwater (cyan/green, depleted
    # red) stays vivid; sky/sunset topside uses natural so warm scenes are not
    # over-boosted. This does not change vivid numerics on underwater frames.
    sample=preview_size(rgb,320) if min(rgb.shape[:2])>64 else rgb
    top=sample[:max(1,sample.shape[0]//3)]
    r,g,b=[float(x) for x in top.reshape(-1,3).mean(0)]
    red_ratio=r/(r+g+b+1e-6)
    hsv=cv2.cvtColor(top,cv2.COLOR_RGB2HSV)
    hue,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    usable=(val>25)&(sat>20)
    if np.any(usable):
        h=hue[usable]
        warm=float(((h<=22)|(h>=165)).mean())
        water=float(((h>=45)&(h<=110)).mean())
        sky=float(((h>110)&(h<=140)).mean())
    else:
        warm=water=sky=0.0
    if warm>0.12 and red_ratio>0.32:
        return 'natural'
    if sky>water and red_ratio>0.28:
        return 'natural'
    if red_ratio<0.30 or water>=sky:
        return 'vivid'
    return 'natural'

def protect_highlights(lab,source):
    # Green-clipped cyan highlights have unreliable recovered red. Reduce their
    # chroma smoothly, avoiding the pink patches produced by independent gains.
    green=source[:,:,1].astype(np.float32)/255
    blue=source[:,:,2].astype(np.float32)/255
    mask=np.clip((green-.88)/.10,0,1)*np.clip((blue-.35)/.35,0,1)
    mask*=np.clip((lab[:,:,0]-55)/25,0,1)
    lab[:,:,1:]*=(1-.95*mask[:,:,None])
    return lab

def particle_mask(rgb):
    # Detect compact bright outliers at half resolution. Restrict repairs to
    # smooth background away from edges; never globally erase bright details.
    h,w=rgb.shape[:2]
    small=cv2.resize(rgb,(w//2,h//2),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    background=cv2.medianBlur(gray,11)
    residual=gray.astype(np.float32)-background
    bg=background.astype(np.float32)
    mu=cv2.boxFilter(bg,-1,(25,25))
    std=np.sqrt(np.maximum(cv2.boxFilter(bg*bg,-1,(25,25))-mu*mu,0))
    edges=cv2.Canny(background,25,60)
    distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    candidates=((residual>17)&(std<5)&(distance>5)).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(candidates,8)
    selected=np.zeros_like(gray)
    accepted=[]
    for label in range(1,n):
        x,y,bw,bh,area=stats[label]
        if not 2<=area<=65 or max(bw,bh)>15 or max(bw,bh)/max(1,min(bw,bh))>2.2:
            continue
        if x<15 or y<15 or x+bw>=gray.shape[1]-15 or y+bh>=gray.shape[0]-15:
            continue
        local=labels[y:y+bh,x:x+bw]==label
        if area/(bw*bh)<.35 or residual[y:y+bh,x:x+bw][local].max()<25:
            continue
        # Avoid partial detections along the edge of a larger bright object.
        cy,cx=round(centroids[label][1]),round(centroids[label][0])
        ring=gray[cy-10:cy+11,cx-10:cx+11].astype(np.float32)
        outer=np.ones((21,21),bool); outer[6:15,6:15]=False
        ring_values=ring[outer]
        if np.std(ring_values)>7 or np.percentile(ring_values,95)-np.median(ring_values)>12:
            continue
        selected[y:y+bh,x:x+bw][local]=255
        accepted.append([int(x*2),int(y*2),int(bw*2),int(bh*2)])
    selected=cv2.dilate(selected,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    full=cv2.resize(selected,(w,h),interpolation=cv2.INTER_NEAREST)
    return full,accepted

def polish(rgb):
    mask,particles=particle_mask(rgb)
    if particles:
        clean=cv2.inpaint(rgb,mask,3,cv2.INPAINT_TELEA)
    else:
        clean=rgb
    small=preview_size(clean,1200)
    sample_lab=cv2.cvtColor(small.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    black=min(8.,.42*float(np.percentile(sample_lab[:,:,0],.5)))
    h,w=rgb.shape[:2]
    result=np.empty_like(rgb)
    # Halo supports the largest blur and prevents visible tile boundaries.
    for start in range(0,h,384):
        lo=max(0,start-64); hi=min(h,start+384+64)
        lab=cv2.cvtColor(clean[lo:hi].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        l=cv2.bilateralFilter(lab[:,:,0],5,1.6,2)
        for channel in (1,2):
            lab[:,:,channel]=cv2.bilateralFilter(lab[:,:,channel],7,3.5,3)
        local=l-cv2.GaussianBlur(l,(0,0),18)
        clarity=.20*np.sign(local)*np.maximum(np.abs(local)-.7,0)
        l=np.clip((l-black)*100/(100-black),0,100)
        wave=np.sin(np.pi*l/100)
        l=l+.12*(l-50)*wave+2.2*wave
        lab[:,:,0]=np.clip(l+clarity,0,99)
        chroma=np.sqrt(lab[:,:,1]**2+lab[:,:,2]**2)
        vibrance=1+.24*np.exp(-chroma/35)
        lab[:,:,1:]*=vibrance[:,:,None]
        out=np.round(lab_to_rgb_safe(lab)*255).astype(np.uint8)
        end=min(h,start+384)
        result[start:end]=out[start-lo:end-lo]
    return result,mask,particles,black

def richer(rgb):
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    chroma=np.hypot(lab[:,:,1],lab[:,:,2])
    # Large chroma increase, tapering for already vivid colors and near-neutrals.
    gain=1+.95*(1-np.exp(-(chroma/6)**2))*np.exp(-chroma/120)
    # Keep very bright near-neutral whites from acquiring an amplified tint.
    white=np.clip((lab[:,:,0]-75)/20,0,1)*np.exp(-(chroma/18)**2)
    gain=1+(gain-1)*(1-.85*white)
    # Amplify residual green less than magenta/warm hues so vivid does not
    # reintroduce the underwater green cast.
    a=lab[:,:,1]
    a_gain=np.where(a>=0,gain,1+(gain-1)*.40)
    lab[:,:,1]=a*a_gain
    lab[:,:,2]*=gain
    return np.round(lab_to_rgb_safe(lab)*255).astype(np.uint8)

def reduce_green_cast(rgb, target_white_a=2.2):
    # Adaptive residual-green cleanup after correction/vivid. Measures cast from
    # bright near-neutrals, nudges whites toward a mild magenta-neutral target,
    # warms olive midtones toward brown, and rotates lime HSV hues toward yellow.
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    L,a,b=lab[:,:,0],lab[:,:,1],lab[:,:,2]
    chroma=np.hypot(a,b)
    mask=(L>65)&(L<93)&(chroma<20)
    if np.count_nonzero(mask)<500:
        mask=(L>58)&(chroma<26)
    if np.count_nonzero(mask)<200:
        # Fall back to midtone near-neutrals when specular whites are scarce.
        mask=(L>35)&(L>0)&(L<75)&(chroma<16)
    if np.count_nonzero(mask)>=150:
        aa,bb=a[mask],b[mask]
        keep=((aa>=np.percentile(aa,20))&(aa<=np.percentile(aa,80))&
              (bb>=np.percentile(bb,20))&(bb<=np.percentile(bb,80)))
        a_cast=float(aa[keep].mean()); b_cast=float(bb[keep].mean())
    else:
        a_cast=float(np.median(a)); b_cast=float(np.median(b))
    w=np.clip((L-5)/16,0,1)*np.clip((98-L)/8,0,1)
    greenish=np.clip((-a)/8,0,1)
    a_needed=max(0.0,min(target_white_a-a_cast,8.5)) if a_cast<target_white_a else 0.0
    lab[:,:,1]=a+a_needed*w*(.80+.20*greenish)
    if a_cast<.8 and b_cast>1.0:
        lab[:,:,2]=lab[:,:,2]-.55*min(b_cast,8.0)*w*greenish
    olive_amt=3.2+3.0*np.clip((target_white_a-a_cast)/6,0,1)
    olive=((lab[:,:,1]>-16)&(lab[:,:,1]<5)&(lab[:,:,2]>5)&(lab[:,:,2]<30)&
           (chroma>5)&(chroma<34)&(L>16)&(L<80)).astype(np.float32)
    olive*=np.clip((10+np.minimum(lab[:,:,1],0))/9,0,1)
    lab[:,:,1]=lab[:,:,1]+olive_amt*olive*w
    lab[:,:,2]=lab[:,:,2]+1.3*olive*w
    still=np.maximum(-lab[:,:,1],0)
    lab[:,:,1]=lab[:,:,1]+.85*still*w
    warmed=np.round(lab_to_rgb_safe(lab)*255).astype(np.uint8)
    # Rotate lime/yellow-green toward golden yellow without crushing true greens
    # that already sit near cyan-blue water hues.
    hsv=cv2.cvtColor(warmed,cv2.COLOR_RGB2HSV).astype(np.float32)
    hue,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    lime=((hue>=28)&(hue<=78)&(sat>28)&(val>35)&(val<245)).astype(np.float32)
    lime*=np.clip((hue-24)/40,0,1)
    # Strength tracks how green the measured neutrals still were.
    lime_shift=(6.0+6.0*np.clip((target_white_a-a_cast)/6,0,1))*lime
    hsv[:,:,0]=np.clip(hue-lime_shift,0,179)
    out=cv2.cvtColor(np.round(hsv).astype(np.uint8),cv2.COLOR_HSV2RGB)
    return out,{'neutral_a_cast':a_cast,'neutral_b_cast':b_cast,
                'magenta_shift':a_needed,'olive_amount':float(olive_amt)}
