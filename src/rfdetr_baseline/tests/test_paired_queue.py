import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest

from train_coco_paired import python_script, is_project_job, wait_for_idle, commands


class PairedQueueTests(unittest.TestCase):
    def test_extracts_actual_script_and_ignores_python_command_strings(self):
        self.assertEqual(python_script(['/env/bin/python', '-u', '/project/train.py', '--config', 'a.json']), '/project/train.py')
        self.assertIsNone(python_script(['python', '-c', 'print("train.py")']))
        self.assertIsNone(python_script(['tail', '-F', 'train.py']))
        self.assertEqual(python_script(['python3.11', '-X', 'utf8', 'evaluate.py']), 'evaluate.py')

    def test_ignores_unrelated_projects_arguments_and_zombies(self):
        root = Path.cwd()/'queue_test_project'
        self.assertTrue(is_project_job(['python', '-u', str(root/'train.py')], root, 'S', root))
        self.assertFalse(is_project_job(['python', str(root/'plot.py'), '--caption', 'train.py'], root, 'S', root))
        self.assertFalse(is_project_job(['python', str(root/'pretrain.py')], root, 'S', root))
        self.assertFalse(is_project_job(['python', str(root.parent/'other/train.py')], root, 'S', root))
        self.assertFalse(is_project_job(['python', str(root/'train.py')], root, 'Z', root))
        self.assertTrue(is_project_job(['python', 'run_both.py'], root, 'T', root))

    def test_no_wait_refuses_without_starting_or_terminating_a_job(self):
        jobs = [{'pid': 123, 'state': 'S', 'argv': ['python', 'train.py']}]
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'occupied'):
            wait_for_idle(False, scanner=lambda: jobs, sleeper=lambda _: self.fail('must not sleep'))

    def test_wait_releases_after_existing_job_finishes(self):
        snapshots = iter([[{'pid': 123, 'state': 'S', 'argv': ['python', 'train.py']}], []])
        sleeps = []
        stream = io.StringIO()
        with redirect_stdout(stream):
            wait_for_idle(True, 30, scanner=lambda: next(snapshots), sleeper=sleeps.append)
        self.assertEqual(sleeps, [30])
        self.assertIn('PID=123', stream.getvalue())
        self.assertIn('starting next stage', stream.getvalue())

    def test_stopped_task_is_reported_as_paused(self):
        snapshots = iter([[{'pid': 123, 'state': 'T', 'argv': ['python', 'run_both.py']}], []])
        stream = io.StringIO()
        with redirect_stdout(stream):
            wait_for_idle(True, scanner=lambda: next(snapshots), sleeper=lambda _: None)
        self.assertIn('stopped/paused', stream.getvalue())

    def test_preserves_two_training_two_evaluation_and_two_scoring_stages(self):
        plan = commands()
        self.assertEqual(len(plan), 6)
        self.assertEqual([gpu for _, _, gpu in plan], [True, True, True, True, False, False])
        self.assertIn('--allow-coco-subset', plan[0][1])
        self.assertNotIn('--allow-coco-subset', plan[2][1])
        self.assertIn('original', plan[4][1])
        self.assertIn('voc20', plan[5][1])


if __name__ == '__main__':
    unittest.main()
