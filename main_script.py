import logging
import os
import subprocess
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import closing, cube
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

import toolbox as omero
from getpass import getpass
import argh

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

TEMP_DIR = '/home/ubuntu/data/mydatalocal'
ILASTIK_PATH = '/opt/ilastik-1.3.3post3-Linux/run_ilastik.sh'
THRESHOLD = 200
CLOSING_DISTANCE = 10
CHANNEL_OF_INTEREST = 0


def load_models(project, path):
    file_paths = []
    for ann in project.listAnnotations():
        if isinstance(ann, omero.gw.FileAnnotationWrapper):
            name = ann.getFile().getName()
            if name.endswith(".ilp"):
                file_path = os.path.join(path, name)
                with open(str(file_path), 'wb') as f:
                    for chunk in ann.getFileInChunks():
                        f.write(chunk)
                file_paths.append(file_path)

    return file_paths


def run_ilastik(ilastik_path, input_path, model_path):

    cmd = [ilastik_path,
           '--headless',
           f'--project={model_path}',
           '--export_source=Probabilities',
           '--output_format=numpy',
           # '--output_filename_format={dataset_dir}/{nickname}_Probabilities.npy',
           '--export_dtype=uint8',
           # '--output_axis_order=zctyx',
           input_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as e:
        print(f'Input command: {cmd}')
        print()
        print(f'Error: {e.output}')
        print()
        print(f'Command: {e.cmd}')
        print()


def analyze_image(image, model):
    raw_intensities = omero.get_intensities(image)
    temp_file = f'{TEMP_DIR}/temp_array.npy'
    np.save(temp_file, raw_intensities.transpose((2, 0, 3, 4, 1)))  # zctyx -> tzyxc

    run_ilastik(ILASTIK_PATH, temp_file, model)

    raw_probabilities = np.load(f'{TEMP_DIR}/temp_array_Probabilities.npy')
    raw_probabilities = raw_probabilities.transpose((1, 4, 0, 2, 3))
    # raw_probabilities = np.squeeze(raw_probabilities)

    thresholded = apply_hysteresis_threshold(raw_probabilities[:, CHANNEL_OF_INTEREST, 0, ...],
                                             low=THRESHOLD * .5,
                                             high=THRESHOLD)

    closed = closing(thresholded, cube(CLOSING_DISTANCE))
    cleared = clear_border(closed)

    labels = label(cleared)

    labels = np.expand_dims(labels, axis=(1, 2))

    regions = regionprops(labels, raw_intensities[:, CHANNEL_OF_INTEREST, 0, ...])
    return regions


def run(user, password, group, dataset_id: int, host='omero.mri.cnrs.fr'):
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=user,
                                     password=password,
                                     host=host,
                                     port=4064,
                                     group=group)
        conn.c.enableKeepAlive(60)

        dataset = omero.get_dataset(conn, dataset_id)
        project = dataset.getParent()
        model_paths = load_models(project, TEMP_DIR)

        images = dataset.listChildren()

        measurements = {'image': [],
                        'roi': [],
                        'area': [],
                        'convex_area': [],
                        'bbox': [],
                        'centroid': [],
                        'weighted_centroid': [],
                        'equivalent_diameter': [],
                        'feret_diameter_max': [],
                        'major_axis_length': [],
                        'minor_axis_length': [],
                        'solidity': [],
                        'max_intensity': [],
                        'mean_intensity': [],
                        'min_intensity': [],
                        }

        for image in images:
            regions = analyze_image(image, model_paths[0])

            for region in regions:
                roi = omero.create_roi(conn, image, [omero.create_shape_point(x_pos=region.centroid[2],
                                                                              y_pos=region.centroid[1],
                                                                              z_pos=region.centroid[0],
                                                                              t_pos=0)
                                                     ]
                                       )
                measurements['image'].append(image)
                measurements['roi'].append(roi)
                for k, v in measurements.items():
                    try:
                        v.append(getattr(region, k))
                    except AttributeError:
                        pass

        table = omero.create_annotation_table(conn,
                                              table_name='Measurements',
                                              column_names=[k for k, _ in measurements.items()],
                                              column_descriptions=[[] for _ in range(len(measurements))],
                                              values=[v for _, v in measurements.items()],
                                              )

    finally:
        conn.close()
        logger.info('Done')


if __name__ == '__main__':
    argh.dispatch_command(run)
