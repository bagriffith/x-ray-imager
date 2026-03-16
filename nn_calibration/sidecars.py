import yaml
import pathlib
import logging
import numpy as np
import nn_calibration
import xy_testing

logger = logging.getLogger(__name__)


def load(sidecar_path):
    yaml_file = pathlib.Path(sidecar_path)
    dat_path = yaml_file.with_suffix('.dat')

    with open(yaml_file) as file:
        run_info = yaml.safe_load(file)

    logger.debug(sidecar_path)
    set_params = {'nice_spacing': not run_info['grid']['even'],
                  'grid': str(run_info['grid']['name']),
                  'duration': round(run_info['time']['duration']),
                  'buffer': round(run_info['time']['pad']),
                  'imager': str(run_info['imager'])}

    if 'source' in run_info:
        set_params['source'] = str(run_info['source'])

    return dat_path, set_params


def sidecar_path(paths, list_files=False):
    files = []
    for f in paths:
        if list_files:
            f_files = nn_calibration.sidecars.grab_list(f)
        else:
            f_files = [f]

        for sub_f in f_files:
            sub_f = pathlib.Path(sub_f)
            if not sub_f.exists():
                raise ValueError(f'{sub_f} not found.')
            if sub_f.is_dir():
                files.extend(sub_f.glob('**/*.yaml'))
                files.extend(sub_f.glob('**/*.yml'))
            else:
                if sub_f.suffix.lower() not in ['.yml', '.yaml']:
                    raise ValueError(f'{sub_f} is not a yaml file.')
                files.append(sub_f)
    return files


def grab_list(list_path):
    target_files = []
    with open(list_path) as f:
        for line in f.readlines():
            path_str = line.strip()
            if path_str[0] == '#':
                continue
            target_files.append(pathlib.Path(path_str))
            assert target_files[-1].exists()
    if len(target_files) == 0:
        raise ValueError('No files found.')
    return target_files


def load_info(sidecar, mask=False, label=False):
    dat_path, run_info = load(sidecar)

    energies = nn_calibration.cluster.optimal_params[run_info['source']]['energy']

    if mask:
        labels = nn_calibration.cluster.optimal_params[run_info['source']]['for_fit']
        energies = [energies[n] for n in labels]
    else:
        labels = list(range(len(energies)))

    if run_info['nice_spacing']:
        grid = xy_testing.grid.nice_spacing[run_info['grid']]
    else:
        grid = xy_testing.grid.even_spacing[run_info['grid']]

    x = np.double(grid.x) - xy_testing.grid.CRYSTAL_WIDTH/2
    y = np.double(grid.y) - xy_testing.grid.CRYSTAL_WIDTH/2
    positions = np.meshgrid(x, y, indexing='ij')

    if label:
        return energies, positions, labels
    else:
        return energies, positions


def load_all_info(sidecar_list, mask=False):
    energies = []
    positions = None

    for f in sidecar_list:
        e, pos = load_info(f, mask)
        pos = np.array(pos)
        energies.extend(e)
        if positions is None:
            positions = pos
        else:
            assert np.all(np.equal(pos, positions))

    return energies, positions
