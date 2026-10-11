"""Watchdog tests: native readiness must never suppress compatibility dialogs."""
import unittest
from unittest.mock import Mock, patch
import photoshop_psd as psd


def window(name, width=260, height=224, layer=8, wid=10):
    return dict(id=wid,name=name,layer=layer,bounds=dict(Width=width,Height=height))


class NativeReadiness(unittest.TestCase):
    def test_readiness_probe_is_read_only(self):
        process=Mock();process.poll.return_value=0
        process.communicate.return_value=('ready\n','');process.returncode=0
        observations=[]
        with patch.object(psd.subprocess,'Popen',return_value=process) as launch, patch.object(psd,'windows',return_value=[]), patch.object(psd.time,'monotonic',return_value=1):
            psd.wait_native_document('/tmp/generated.psd',set(),observations,0,10)
        script=launch.call_args.args[0][-1]
        self.assertIn('app.documents',script)
        for mutation in ['displayDialogs','preferences','close(', 'save(', 'open(']:
            self.assertNotIn(mutation,script)
        self.assertEqual(observations[0]['stage'],'native-open-readiness')
        process.terminate.assert_not_called()

    def test_warning_aborts_probe_before_inspection(self):
        process=Mock();process.poll.return_value=None
        observations=[]
        with patch.object(psd.subprocess,'Popen',return_value=process), patch.object(psd,'windows',return_value=[window('Adobe Photoshop')]), patch.object(psd.time,'monotonic',return_value=1):
            with self.assertRaisesRegex(RuntimeError,'before inspection'):
                psd.wait_native_document('/tmp/generated.psd',set(),observations,0,10)
        process.terminate.assert_called_once()
        self.assertEqual(len(observations[0]['dialogs']),1)

    def test_blocked_probe_remains_bounded(self):
        process=Mock();process.poll.return_value=None
        with patch.object(psd.subprocess,'Popen',return_value=process), patch.object(psd,'windows',return_value=[]), patch.object(psd.time,'monotonic',return_value=11):
            with self.assertRaises(TimeoutError):
                psd.wait_native_document('/tmp/generated.psd',set(),[],0,10)
        process.terminate.assert_called_once()

    def test_pixel_toolbar_requires_document_and_observed_geometry(self):
        document=window('generated.psd @ 100%',1728,1024,0,1)
        toolbar=window('Pixel Layer Bar',509,32)
        self.assertEqual(psd.dialog_windows([document,toolbar],set()),[])
        self.assertEqual(psd.dialog_windows([toolbar],set()),[toolbar])
        malformed=window('Pixel Layer Bar',509,224)
        self.assertEqual(psd.dialog_windows([document,malformed],set()),[malformed])

    def test_toolbar_exemption_does_not_hide_native_alerts(self):
        document=window('generated.psd @ 100%',1728,1024,0,1)
        alert=window('Adobe Photoshop',260,224,wid=2)
        toolbar=window('Pixel Layer Bar',539,32,wid=3)
        self.assertEqual(psd.dialog_windows([document,toolbar,alert],set()),[alert])


if __name__=='__main__':
    unittest.main()
