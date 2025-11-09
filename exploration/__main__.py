import os

import pandas as pd
import numpy as np

from exploration.site import Site, Well, WellSegment
from config import settings


def main(voxel_size: int, inputfile_path: str, outputfile_path: str, advanced: bool = False):
    site = Site(voxel_size)

    wellheads_data = pd.read_excel(inputfile_path, 0, usecols=range(4))
    wellheads_data.columns = ['name', 'x', 'y', 'z']

    wellsegments_data = pd.read_excel(inputfile_path, 1, usecols=range(4))
    wellsegments_data.columns = ['well_name', 'azimuth', 'zenith', 'depth']
    wellsegments_data['depth_lag'] = wellsegments_data.groupby(['well_name'])['depth'].shift(1)
    wellsegments_data = wellsegments_data[~wellsegments_data['depth_lag'].isna()]

    lithology_data = pd.read_excel(inputfile_path, 2, usecols=[0, 2, 4])
    lithology_data.columns = ['well_name', 'depth', 'code']
    lithology_data['depth_lag'] = lithology_data.groupby(['well_name'])['depth'].shift(1, fill_value=0)

    for i, wellhead_data in wellheads_data.iterrows():
        well_name = wellhead_data['name']
        well = Well(well_name, (wellhead_data['x'], wellhead_data['y'], wellhead_data['z']))

        previous_segment_end = np.array([0, 0, 0])
        for ii, segment_data in wellsegments_data[wellsegments_data['well_name'] == well_name].iterrows():
            segment_start = segment_data['depth_lag']
            segment_end = segment_data['depth']

            lithology_slice = lithology_data[
                (lithology_data['well_name'] == well_name)
                & (
                    (lithology_data['depth_lag'] <= segment_end)
                    & (lithology_data['depth'] >= segment_start)
                )
            ]

            if lithology_slice.empty:
                lithology_slice = lithology_data[
                    (lithology_data['well_name'] == well_name)
                    & (lithology_data['depth_lag'] >= segment_start)
                ][:1]

            for iii, lithology_datum in lithology_slice.iterrows():
                length = min(segment_end, lithology_datum['depth']) - max(segment_start, lithology_datum['depth_lag'])
                if not length:
                    continue

                segment = WellSegment.from_spherical(
                    previous_segment_end,
                    segment_data['azimuth'],
                    segment_data['zenith'],
                    length
                )
                segment.content = lithology_datum['code']

                previous_segment_end = segment.end

                well.add_segment(segment)

        site.add_well(well)

    # Выбор метода обработки: упрощенный по умолчанию, расширенный при флаге --advanced
    if advanced:
        df = site.process_advanced(normalize='minmax')
    else:
        df = site.process_simple()
    
    df.to_csv(outputfile_path, index=False, encoding='utf-8')


if __name__ == '__main__':

    cfg = settings.section("exploration")
    input_path = cfg.get("input_file_path")
    output_path = cfg.get("output_file_path", "./temp/explored_blocks.csv")
    voxel_size = int(cfg.get("block_size", 1))
    advanced = bool(cfg.get("advanced", False))

    main(voxel_size, input_path, output_path, advanced)
    print(f'Done, saved to {os.path.abspath(output_path)}')
