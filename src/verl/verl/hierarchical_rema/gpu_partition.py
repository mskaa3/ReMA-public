"""Partition Slurm's GPU mask before starting Ray or importing torch in the driver.

Run as a file, not with -m: importing the package first could initialize CUDA.
The symmetric per-node reservation keeps Ray generation and learner pools equal.
"""
import argparse
import os


def partition(visible, reserve, role):
    devices = [value.strip() for value in visible.split(',') if value.strip()]
    if reserve == 0:
        return visible
    if reserve != 1:
        raise ValueError('The live reward pilot reserves exactly one GPU per node')
    if len(devices) < 2 or len(devices) != len(set(devices)) or '-1' in devices:
        raise ValueError('Need at least two distinct allocated GPUs in CUDA_VISIBLE_DEVICES; refusing GPU sharing')
    if role not in ('policy', 'reward'):
        raise ValueError('Unknown GPU role')
    return ','.join(devices[:-1] if role == 'policy' else devices[-1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['policy', 'reward'], required=True)
    parser.add_argument('--reserve', type=int, default=1)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    env = os.environ.copy()
    if args.reserve:
        try:
            mask = partition(env.get('CUDA_VISIBLE_DEVICES', ''), args.reserve, args.role)
        except ValueError as exc:
            parser.error(str(exc))
        env['CUDA_VISIBLE_DEVICES'] = mask
        env['REMA_GPU_ROLE'] = args.role
        print(f'[rema-gpu] role={args.role} CUDA_VISIBLE_DEVICES={mask}', flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
