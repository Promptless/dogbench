import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from dogbench.mirror import validate_prepared_mirror


def git(path, *args):
    return subprocess.check_output(['git','-C',str(path),*args], text=True).strip()


class MirrorBoundaryTests(unittest.TestCase):
    def test_pinned_tree_and_explicit_fetch_push_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prepared=root/'prepared'; prepared.mkdir()
            git(prepared,'init','--quiet','--initial-branch=main')
            git(prepared,'config','user.name','Test'); git(prepared,'config','user.email','test@example.invalid')
            (prepared/'guide.md').write_text('frozen docs\n')
            git(prepared,'add','.');git(prepared,'commit','--quiet','-m','snapshot')
            mirror=root/'mirror'
            subprocess.run(['git','clone','--quiet',str(prepared),str(mirror)],check=True)
            git(mirror,'remote','set-url','origin','https://github.com/owned/task.git')
            sha=git(mirror,'rev-parse','HEAD')
            result=validate_prepared_mirror(mirror,'owned/task',sha,prepared)
            self.assertEqual(result['docs_tree'],git(prepared,'rev-parse','HEAD^{tree}'))
            git(mirror,'remote','set-url','--push','origin','git@github.com:other/task.git')
            with self.assertRaisesRegex(RuntimeError,'fetch/push'):
                validate_prepared_mirror(mirror,'owned/task',sha,prepared)
            git(mirror,'remote','set-url','--push','origin','git@github.com:owned/task.git')
            (mirror/'guide.md').write_text('modified\n')
            with self.assertRaisesRegex(RuntimeError,'clean'):
                validate_prepared_mirror(mirror,'owned/task',sha,prepared)


if __name__ == '__main__':
    unittest.main()


class ManagedMirrorBoundaryTests(unittest.TestCase):
    def test_original_managed_transform_and_code_blob_attestation(self):
        from dogbench.cloud_mirrors import Mirror, configure_mirrors
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prepared=root/'prepared'; prepared.mkdir()
            git(prepared,'init','--quiet','--initial-branch=main')
            git(prepared,'config','user.name','Test');git(prepared,'config','user.email','test@example.invalid')
            (prepared/'docs').mkdir();(prepared/'docs/guide.md').write_text('Guide\n')
            (prepared/'mkdocs.yml').write_text('site_name: Docs\nnav:\n  - Guide: guide.md\n')
            (prepared/'.github').mkdir();(prepared/'.github/automation.yml').write_text('workflow\n')
            (prepared/'package.json').write_text('{"repository":"https://github.com/upstream/project","value":1}\n')
            git(prepared,'add','.');git(prepared,'commit','--quiet','-m','snapshot')
            code=root/'code';subprocess.run(['git','clone','--quiet',str(prepared),str(code)],check=True)
            git(code,'config','user.name','Test');git(code,'config','user.email','test@example.invalid')
            (code/'package.json').write_text('{"repository":"https://github.com/upstream/project","value":2}\n')
            git(code,'add','.');git(code,'commit','--quiet','-m','change')
            target=root/'mirror';target.mkdir();origin=root/'origin.git';origin.mkdir()
            git(target,'init','--quiet','--initial-branch=main');git(origin,'init','--quiet','--bare','--initial-branch=main')
            git(target,'remote','add','origin',str(origin))
            overlays=root/'overlays';sources=root/'source-overlays';overlays.mkdir();sources.mkdir()
            configure_mirrors(index_path=root/'index.json',mirror_overlays_root=overlays,source_overlays_root=sources)
            mirror=Mirror('upstream/project','owned/neutral',prepared,target,'')
            base=mirror.set_to_sha(git(prepared,'rev-parse','HEAD'))
            self.assertFalse((target/'.github').exists())
            self.assertTrue((target/'docs.json').is_file())
            git(target,'remote','set-url','origin','https://github.com/owned/neutral.git')
            details=validate_prepared_mirror(target,'owned/neutral',base,prepared,
                prepared_code_dir=code,managed_transport=True,source_repo='upstream/project',
                code_source_repo='upstream/project',mirror_overlays_root=overlays,source_overlays_root=sources)
            self.assertEqual(details['transport_mode'],'managed')
            self.assertEqual(details['code_paths'],['package.json'])
            self.assertEqual(details['code_base_blob_oids']['package.json'],git(target,'rev-parse','HEAD:package.json'))
            self.assertNotEqual(details['code_head_blob_oids']['package.json'],git(code,'rev-parse','HEAD:package.json'))
            (overlays/'neutral').mkdir();(overlays/'neutral/.mintignore').write_text('changed-after-publication\n')
            with self.assertRaisesRegex(RuntimeError,'differs'):
                validate_prepared_mirror(target,'owned/neutral',base,prepared,managed_transport=True,
                    source_repo='upstream/project',mirror_overlays_root=overlays,source_overlays_root=sources)
