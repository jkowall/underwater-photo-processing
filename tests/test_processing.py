import json
import sys
import tempfile
import unittest
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
        with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out)]):
            with self.assertRaises(SystemExit) as error:runner.main()
        self.assertEqual(error.exception.code,2)
        self.assertEqual(self.dest.read_bytes(),original)
        self.assertFalse((self.out/'.processing.lock').exists())

    def test_existing_lock_is_not_removed(self):
        (self.out/'.processing.lock').mkdir()
        with patch.object(sys,'argv',['process_batch','--input',str(self.src),'--output',str(self.out)]):
            with self.assertRaises(SystemExit):runner.main()
        self.assertTrue((self.out/'.processing.lock').exists())


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
