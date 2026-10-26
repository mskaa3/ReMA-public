"""CPU regression tests for plan-aware selection and exact C3 prefix sharing."""

import ast
import hashlib
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PPO = ROOT / 'src/verl/verl/rema_separated_trainer/ppo'
sys.path.insert(0, str(PPO))
from c3_role_selection import select_present_worker_role
from local_results import extract_local_result
import experiment_io


class RoleSelectionTests(unittest.TestCase):
    def test_counts_distinct_prefixes_not_32_suffix_copies(self):
        groups = ['a'] * 32 + ['b'] * 32
        plans = {i: [('w1', '', [])] + ([('w2', '', [])] if i < 32 else [])
                 for i in range(64)}
        rng = Mock()
        rng.choice.return_value = 'w2'
        selected, counts = select_present_worker_role(plans, groups, ['w1', 'w2', 'w3'], rng)
        self.assertEqual(selected, 'w2')
        self.assertEqual(counts, {'w1': 2, 'w2': 1, 'w3': 0})
        rng.choice.assert_called_once_with(['w1', 'w2'], p=[2 / 3, 1 / 3])

    def test_empty_plans_do_not_invent_a_worker(self):
        rng = Mock()
        self.assertEqual(select_present_worker_role({0: []}, ['a'], ['w1'], rng),
                         (None, {'w1': 0}))
        rng.choice.assert_not_called()

    def test_inconsistent_shared_plans_fail(self):
        with self.assertRaisesRegex(ValueError, 'shared plan'):
            select_present_worker_role({0: [('w1', '', [])], 1: []}, ['a', 'a'], ['w1'], Mock())

    def test_real_rollout_selects_before_branching_and_keeps_eight_by_four(self):
        path = PPO / 'multi_agent_rollout.py'
        definitions = [n for n in ast.parse(path.read_text()).body
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        tree = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
        ast.fix_missing_locations(tree)
        helpers = runpy.run_path(str(PPO / 'c3_role_selection.py'))
        for focal in ('worker_stage_1', 'worker_stage_2'):
            with self.subTest(focal=focal):
                rng = Mock()
                rng.choice.return_value = focal
                ns = dict(np=np, re=re, hashlib=hashlib, unicodedata=unicodedata,
                          extract_local_result=extract_local_result)
                ns['select_present_worker_role'] = lambda plans, groups, roles, _: helpers[
                    'select_present_worker_role'](plans, groups, roles, rng)
                probe_tree = ast.parse((PPO / 'validation_probes.py').read_text())
                make_inputs = next(n for n in probe_tree.body if isinstance(n, ast.FunctionDef)
                                   and n.name == 'make_probe_inputs')
                exec(compile(ast.Module(body=[make_inputs], type_ignores=[]), '<probe>', 'exec'), ns)
                exec(compile(tree, str(path), 'exec'), ns)
                rollout = object.__new__(ns['MultiAgentRollout'])
                rollout.config = SimpleNamespace(stop_when_truncated=True)
                stages = [f'worker_stage_{i}' for i in range(1, 5)]
                roles = ['decomposer', 'selector', *stages]
                hierarchy = dict(stage_roles=stages, worker_roles=['general_math_worker'],
                    routing_mode='sequential_plan', max_planned_subtasks=4, terminal_worker_as_answer=True,
                    scoped_c3_grpo={'enable': True, 'branch_turn': 0, 'continuations_per_action': 4})
                calls = []
                token = 0

                def generate(role, chats, *args, **kwargs):
                    nonlocal token
                    calls.append((role, len(chats)))
                    outputs, ids = [], []
                    for _ in chats:
                        token += 1
                        outputs.append('PLAN:\n- S1: Compute x.\n- S2: Use S1.' if role == 'decomposer'
                                       else f'REASONING: Compute.\nLOCAL_RESULT: \\boxed{{{token}}}')
                        ids.append([token])
                    return outputs, [1] * len(chats), ['stop'] * len(chats), None, ids

                rollout._generate_from_chat_list = generate
                prompts = SimpleNamespace(non_tensor_batch={'question': ['q'] * 32, 'uid': ['a'] * 32},
                    meta_info={'validate': False, 'c3_focal_role': 'worker_stage_4',
                               'c3_select_available_worker': True, 'c3_candidate_worker_roles': stages})
                system = runpy.run_path(str(ROOT / 'prompt/math/hierarchical_mamrp.py'))[
                    'build_hierarchical_system_prompts'](stages)
                histories, flags, reasons = rollout._initialize_conversation_state(32)
                with patch.dict(sys.modules, {'verl.rema_separated_trainer.ppo.experiment_io': experiment_io}):
                    _, _, records, _ = rollout._run_hierarchical_conversation(
                        prompts, {}, 1, roles, system, hierarchy, histories, flags, reasons, 1024, None)
                self.assertEqual(prompts.meta_info['c3_focal_role'], focal)
                self.assertFalse(prompts.meta_info['c3_select_available_worker'])
                expected = [('decomposer', 1), ('worker_stage_1', 8), ('worker_stage_2', 32)]
                if focal == 'worker_stage_2':
                    expected = [('decomposer', 1), ('worker_stage_1', 1), ('worker_stage_2', 8)]
                self.assertEqual(calls, expected)
                self.assertEqual(len({tuple(r['token_ids']) for r in records}), 8)
                for start in range(0, 32, 4):
                    self.assertEqual(len({tuple(r['token_ids']) for r in records[start:start + 4]}), 1)
                    self.assertTrue(all(r['role'] == focal for r in records[start:start + 4]))


if __name__ == '__main__':
    unittest.main()
