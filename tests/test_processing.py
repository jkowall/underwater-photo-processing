import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import process_batch as runner
import underwater_pipeline as pipeline
import install_skill


class ImageTests(unittest.TestCase):
    def test_uniform_images_are_finite(self):
        for value in (0,80,255):
            rgb=np.full((64,96,3),value,np.uint8)
            params=pipeline.estimate(rgb)
            result=pipeline.color_correct(rgb,params)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.isfinite(params['white_balance_gains_RGB']).all())

    def test_vivid_increases_color_without_clipping_whites(self):
        rgb=np.zeros((40,60,3),np.uint8)
        rgb[:,:20]=[180,130,90]
        rgb[:,20:40]=[85,125,185]
        rgb[:,40:]=[235,235,235]
        result=pipeline.richer(rgb)
        before=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        after=cv2.cvtColor(result.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        before_c=np.hypot(before[:,:40,1],before[:,:40,2]).mean()
        after_c=np.hypot(after[:,:40,1],after[:,:40,2]).mean()
        self.assertGreater(after_c,before_c*1.3)
        self.assertLess(int(result.max()),255)
        self.assertLessEqual(int(np.ptp(result[:,40:].astype(int),axis=2).max()),1)

    def test_particle_cleanup_is_local_and_preserves_an_edge(self):
        rgb=np.full((256,256,3),70,np.uint8)
        cv2.circle(rgb,(80,80),5,(200,200,200),-1)
        rgb[:,200:]=200
        mask,boxes=pipeline.particle_mask(rgb)
        self.assertTrue(boxes)
        self.assertGreater(mask[80,80],0)
        self.assertEqual(int(mask[:,194:207].max()),0)
        self.assertLess(np.count_nonzero(mask)/mask.size,.01)

    def test_auto_look_classifies_water_versus_sunset(self):
        water=np.full((80,120,3),[30,110,140],np.uint8)
        sunset=np.zeros((80,120,3),np.uint8)
        sunset[:40]=[220,90,40]
        sunset[40:]=[40,70,160]
        self.assertEqual(pipeline.classify_look(water),'vivid')
        self.assertEqual(pipeline.classify_look(sunset),'natural')

    def test_mild_pre_denoise_preserves_shape_and_chroma_mean(self):
        rng=np.random.default_rng(0)
        rgb=np.clip(np.full((64,96,3),90,np.float32)+rng.normal(0,12,(64,96,3)),0,255).astype(np.uint8)
        out=pipeline.mild_pre_denoise(rgb)
        self.assertEqual(out.shape,rgb.shape)
        before=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        after=cv2.cvtColor(out,cv2.COLOR_RGB2LAB).astype(np.float32)
        # L should smooth slightly; a/b means stay close (no global cast shift)
        self.assertLess(float(after[:,:,0].std()),float(before[:,:,0].std())+1e-3)
        self.assertLess(abs(float(after[:,:,1].mean())-float(before[:,:,1].mean())),1.5)
        self.assertLess(abs(float(after[:,:,2].mean())-float(before[:,:,2].mean())),1.5)

    def test_guided_upsample_matches_guide_size(self):
        guide=np.zeros((120,160,3),np.uint8)
        guide[:,:80]=[40,90,120]
        guide[:,80:]=[180,140,90]
        # Hard vertical edge in guide
        small=cv2.resize(guide,(40,30),interpolation=cv2.INTER_AREA)
        small=cv2.GaussianBlur(small,(0,0),1.2)
        up=pipeline.guided_upsample(small,guide)
        self.assertEqual(up.shape,guide.shape)
        # Edge energy along the mid column should not collapse vs plain Lanczos
        plain=cv2.resize(small,(160,120),interpolation=cv2.INTER_LANCZOS4)
        mid=up.shape[1]//2
        guided_edge=float(np.abs(up[:,mid].astype(np.float32)-up[:,mid-1].astype(np.float32)).mean())
        plain_edge=float(np.abs(plain[:,mid].astype(np.float32)-plain[:,mid-1].astype(np.float32)).mean())
        self.assertGreaterEqual(guided_edge,plain_edge*0.85)

    def test_polish_neural_reports_cleanup_and_is_milder_than_polish(self):
        rgb=np.full((256,256,3),70,np.uint8)
        cv2.circle(rgb,(80,80),5,(200,200,200),-1)
        rgb[:,200:]=[90,130,150]
        mild,mask,boxes,black=pipeline.polish_neural(rgb)
        full,_,_,_=pipeline.polish(rgb)
        self.assertTrue(boxes)
        self.assertGreater(mask[80,80],0)
        self.assertIsInstance(black,float)
        mild_lab=cv2.cvtColor(mild.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        full_lab=cv2.cvtColor(full.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        mild_c=float(np.hypot(mild_lab[:,:,1],mild_lab[:,:,2]).mean())
        full_c=float(np.hypot(full_lab[:,:,1],full_lab[:,:,2]).mean())
        self.assertLess(mild_c,full_c)
        cleaned,green=pipeline.reduce_green_cast(mild)
        self.assertIn('neutral_a_cast',green)
        self.assertEqual(cleaned.shape,rgb.shape)

    def test_rejects_smaller_embedded_preview(self):
        ok,data=cv2.imencode('.jpg',np.zeros((32,48,3),np.uint8))
        self.assertTrue(ok)
        raw=SimpleNamespace(sizes=SimpleNamespace(flip=0,crop_width=96,crop_height=64),
                            extract_thumb=lambda:SimpleNamespace(format=pipeline.rawpy.ThumbFormat.JPEG,data=data.tobytes()))
        with patch.object(pipeline.rawpy,'imread') as opened:
            opened.return_value.__enter__.return_value=raw
            with self.assertRaisesRegex(ValueError,'differs from RAW crop'):
                pipeline.load_source(Path('synthetic.NEF'))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.src=self.root/'source.NEF';self.src.write_bytes(b'source bytes')
        self.out=self.root/'out';self.out.mkdir()
        (self.out/'reports').mkdir();(self.out/'previews').mkdir()
        self.dest=self.out/'source_vivid.png';self.dest.write_bytes(b'verified output stand-in')
        (self.out/'previews'/'source.jpg').write_bytes(b'preview')
        self.record={'recipe_id':'recipe','source_sha256':runner.sha256(self.src),
                     'output_sha256':runner.sha256(self.dest)}
        runner.write_json(self.out/'reports'/'source.json',self.record)

    def valid(self):return runner.resume_valid(self.src,self.out,'vivid','recipe')

    def test_same_source_recipe_and_output_can_resume(self):
        self.assertTrue(self.valid())

    def test_source_or_output_changes_prevent_resume(self):
        self.src.write_bytes(b'changed source');self.assertFalse(self.valid())
        self.src.write_bytes(b'source bytes')
        self.dest.write_bytes(b'changed output');self.assertFalse(self.valid())

    def test_recipe_change_and_partial_report_prevent_resume(self):
        self.assertFalse(runner.resume_valid(self.src,self.out,'vivid','new recipe'))
        (self.out/'reports'/'source.json').write_text('{broken')
        self.assertFalse(self.valid())

    def test_collision_does_not_overwrite_files_or_leave_lock(self):
        original=self.dest.read_bytes()
        with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out),'--look','vivid']):
            with self.assertRaises(SystemExit) as error:runner.main()
        self.assertEqual(error.exception.code,2)
        self.assertEqual(self.dest.read_bytes(),original)
        self.assertFalse((self.out/'.processing.lock').exists())

    def test_existing_lock_is_not_removed(self):
        (self.out/'.processing.lock').mkdir()
        with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out),'--look','vivid']):
            with self.assertRaises(SystemExit):runner.main()
        self.assertTrue((self.out/'.processing.lock').exists())

    def test_stale_lock_mentions_inactive_pid(self):
        lock=self.out/'.processing.lock';lock.mkdir()
        (lock/'pid').write_text('99999999',encoding='utf-8')
        stderr=StringIO()
        with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out),'--look','vivid']):
            with patch.object(sys,'stderr',stderr):
                with self.assertRaises(SystemExit):runner.main()
        self.assertIn('99999999 is not running',stderr.getvalue())
        self.assertTrue(lock.exists())

    def test_resume_skips_verified_output(self):
        self.record['recipe_id']=runner.recipe_id('vivid')
        runner.write_json(self.out/'reports'/'source.json',self.record)
        with patch.object(runner,'process') as processed:
            with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out),'--look','vivid','--resume']):
                self.assertFalse(runner.main())
        processed.assert_not_called()
        summary=json.loads((self.out/'batch_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['resumed'],1)
        self.assertEqual(summary['processed'],0)


class CliTests(unittest.TestCase):
    def test_default_output_is_sibling_look_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            src=root/'Day4 onwards';src.mkdir()
            (src/'a.NEF').write_bytes(b'NEF')
            self.assertEqual(runner.default_output_path(src,'vivid'),root/'Day4 onwards-vivid')
            self.assertEqual(runner.default_output_path(src,'spectroformer'),root/'Day4 onwards-spectroformer')
            with patch.object(runner,'process',return_value={'name':'a_vivid.png','dimensions':[1,1],'bytes':10}):
                with patch.object(sys,'argv',['process_batch','--input',str(src),'--look','vivid']):
                    self.assertFalse(runner.main())
            dest=root/'Day4 onwards-vivid'
            self.assertTrue((dest/'batch_summary.json').is_file())
            self.assertFalse((dest/'.processing.lock').exists())
            with patch.object(runner,'process',return_value={'name':'a_spectroformer.png','dimensions':[1,1],'bytes':10}):
                with patch.object(runner,'check_neural_prereqs'):
                    with patch.object(sys,'argv',['process_batch','--input',str(src)]):
                        self.assertFalse(runner.main())
            dest_auto=root/'Day4 onwards-auto'
            self.assertTrue((dest_auto/'batch_summary.json').is_file())
            self.assertFalse((dest_auto/'.processing.lock').exists())
            with patch.object(runner,'process',return_value={'name':'a_spectroformer.png','dimensions':[1,1],'bytes':10}):
                with patch.object(runner,'check_neural_prereqs'):
                    with patch.object(sys,'argv',['process_batch','--input',str(src),'--look','spectroformer']):
                        self.assertFalse(runner.main())
            dest_sf=root/'Day4 onwards-spectroformer'
            self.assertTrue((dest_sf/'batch_summary.json').is_file())
            self.assertFalse((dest_sf/'.processing.lock').exists())
    def test_format_duration(self):
        self.assertEqual(runner.format_duration(12),'12s')
        self.assertEqual(runner.format_duration(130),'2m10s')
        self.assertEqual(runner.format_duration(3661),'1h01m01s')


class InstallTests(unittest.TestCase):
    def test_skill_payload_and_explicit_updates(self):
        source=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'skill'
            install_skill.install(source,dest)
            self.assertTrue((dest/'scripts'/'process_batch.py').is_file())
            self.assertFalse((dest/'tests').exists())
            self.assertFalse((dest/'README.md').exists())
            with self.assertRaises(FileExistsError):install_skill.install(source,dest)
            (dest/'custom.txt').write_text('keep')
            install_skill.install(source,dest,update=True)
            self.assertEqual((dest/'custom.txt').read_text(),'keep')


if __name__=='__main__':unittest.main()
