#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Description:
-------------------
Splits a binary lesion mask into individual lesion labels using the watershed
algorithm. Seeds (lesion centres) are detected from an energy map built from
optional multimodal images (FLAIR, T1, QSM) and a distance transform. The
resulting label image assigns a unique integer to each separated lesion.

Features:
---------
- Builds a seed map and an energy map from FLAIR, T1, and/or QSM.
- Falls back to a pure distance-transform approach when no modality is given.
- Optionally saves the detected seed/centre map.
- Outputs an integer label image (one label per separated lesion).
- Optionally saves individual binary masks per lesion + seed to a directory.
- Absorbs small watershed fragments into the nearest larger label.

Usage:
------
    run_watershed --mask lesion_mask.nii.gz --out labels.nii.gz
                  --flair flair.nii.gz --t1 t1.nii.gz --qsm qsm.nii.gz

Parameters:
-----------
--mask            : Binary lesion mask (required).
--out             : Output label image (one integer per lesion).
--out_seeds       : (Optional) Output image of detected seed voxels.
--flair           : (Optional) FLAIR image; bright signal drives seed detection.
--t1              : (Optional) T1 image; dark signal (black holes) drives seeds.
--qsm             : (Optional) QSM image; positive iron drives the energy map.
--alpha           : Weight of QSM contribution to the energy map. [0.7]
--beta            : Weight of distance transform subtracted from energy map. [0.3]
--dist            : Minimum distance (voxels) between seed peaks. [3]
--w_flair         : Weight of FLAIR contribution to the seed map. [1.0]
--w_t1            : Weight of T1 contribution to the seed map. [1.5]
--distance_mode   : Distance transform mode: 'euclidean', 'geodesic', or
                    'anisotropic'. [euclidean]
--geodesic_lambda : Strength of the gradient/directional penalty for
                    'geodesic' and 'anisotropic' modes. [1.0]
--blend           : Mix between Euclidean (0) and non-Euclidean (1) distance.
                    0.5 gives equal weight to both. Ignored for 'euclidean'. [0.0]
--min_voxels      : Absorb labels smaller than this into the nearest larger
                    label. 0 = disabled. [6]
--out_lesions_dir : (Optional) Directory where each lesion is saved as an
                    individual binary mask and seed:
                      <out_lesions_dir>/<blob_id>_lesion_<id>.nii.gz
                      <out_lesions_dir>/<blob_id>_seed_<id>.nii.gz

---------------------------------------------------------------
Reference:
[1] Meyer, F. (1994). Topographic distance and watershed lines.
    Signal Processing, 38(1):113-125.
---------------------------------------------------------------
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from LRP.io.utils import (add_overwrite_arg, add_verbose_arg,
                               assert_inputs_exist, assert_outputs_exist)
from LRP.image.labels import (watershed_lesions, absorb_small_confluant_labels)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--in_mask', required=True,
                   help='Binary lesion mask (.nii.gz).')
    p.add_argument('--out_labels', required=True,
                   help='Output label image (.nii.gz).')
    p.add_argument('--out_seeds',
                   help='(Optional) Output image of detected seed voxels.')

    p.add_argument('--flair',
                   help='(Optional) FLAIR image; bright signal drives seed detection.')
    p.add_argument('--qsm',
                   help='(Optional) QSM image; positive iron drives the energy map.')
    p.add_argument('--t1',
                   help='(Optional) T1 image; dark signal (black holes) drives seeds.')

    p.add_argument('--alpha', type=float, default=0.7,
                   help='Weight of QSM contribution to the energy map. [%(default)s]')
    p.add_argument('--beta', type=float, default=0.3,
                   help='Weight of distance transform subtracted from energy map. [%(default)s]')
    p.add_argument('--dist', type=int, default=3,
                   help='Minimum distance (voxels) between seed peaks. [%(default)s]')
    p.add_argument('--w_flair', type=float, default=1.0,
                   help='Weight of FLAIR contribution to the seed map. [%(default)s]')
    p.add_argument('--w_t1', type=float, default=1.5,
                   help='Weight of T1 contribution to the seed map. [%(default)s]')
    p.add_argument('--distance_mode', choices=['euclidean', 'geodesic', 'anisotropic'],
                   default='euclidean',
                   help='Distance transform used to locate lesion centres.\n'
                        '"geodesic"    : isotropic gradient-weighted cost.\n'
                        '"anisotropic" : direction-dependent cost; crossing\n'
                        '                gradients is expensive, moving along\n'
                        '                them is free.\n'
                        'Image priority: QSM > FLAIR > T1. [%(default)s]')
    p.add_argument('--geodesic_lambda', type=float, default=1.0,
                   help='Strength of the gradient/directional penalty.\n'
                        'Used with --distance_mode geodesic or anisotropic. [%(default)s]')
    p.add_argument('--blend', type=float, default=0.0,
                   help='Mix between Euclidean (0.0) and non-Euclidean (1.0)\n'
                        'distance maps. 0.5 = equal weight to both.\n'
                        'Ignored when --distance_mode euclidean. [%(default)s]')
    p.add_argument('--min_voxels', type=int, default=6,
                   help='Absorb watershed labels smaller than this into the\n'
                        'nearest larger label. 0 = disabled. [%(default)s]')

    p.add_argument('--out_lesions_dir',
                   help='(Optional) Directory where each lesion is saved as\n'
                        'an individual binary mask + seed:\n'
                        '  <out_lesions_dir>/<blob_id>_lesion_<id>.nii.gz\n'
                        '  <out_lesions_dir>/<blob_id>_seed_<id>.nii.gz')

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.verbose))

    assert_inputs_exist(parser, [args.in_mask], [args.flair, args.t1, args.qsm])
    assert_outputs_exist(parser, args, [args.out_labels], [args.out_seeds])

    if args.out_lesions_dir:
        os.makedirs(args.out_lesions_dir, exist_ok=True)

    img_ref = nib.load(args.in_mask)
    mask    = img_ref.get_fdata() > 0.5
    flair   = nib.load(args.flair).get_fdata() if args.flair else None
    t1      = nib.load(args.t1).get_fdata()    if args.t1    else None
    qsm     = nib.load(args.qsm).get_fdata()   if args.qsm   else None

    logging.info('Running watershed (distance_mode={})...'.format(args.distance_mode))
    labels, markers = watershed_lesions(mask, alpha=args.alpha, beta=args.beta,
                                        min_distance=args.dist,
                                        w_flair=args.w_flair, w_t1=args.w_t1,
                                        distance_mode=args.distance_mode,
                                        geodesic_lambda=args.geodesic_lambda,
                                        blend=args.blend,
                                        flair=flair, t1=t1, qsm=qsm)

    if args.min_voxels > 0:
        labels = absorb_small_confluant_labels(labels, min_voxels=args.min_voxels)
        logging.info('Small confluant label removal done (min_voxels={}).'.format(args.min_voxels))

    nib.save(nib.Nifti1Image(labels, img_ref.affine, img_ref.header), args.out_labels)
    logging.info('{} lesions detected. Filled voxels: {}/{}'.format(
        labels.max(), np.sum(labels > 0), np.sum(mask)))

    if args.out_seeds:
        nib.save(nib.Nifti1Image(markers, img_ref.affine, img_ref.header), args.out_seeds)
        logging.info('Seed map saved to: {}'.format(args.out_seeds))

    if args.out_lesions_dir:
        blob_id = os.path.splitext(os.path.basename(args.in_mask))[0]
        blob_id = blob_id.replace('.nii', '')
        n_saved = 0
        for lesion_id in range(1, labels.max() + 1):
            lesion_mask = (labels == lesion_id).astype(np.uint8)
            if not lesion_mask.any():
                continue
            nib.save(
                nib.Nifti1Image(lesion_mask, img_ref.affine, img_ref.header),
                os.path.join(args.out_lesions_dir,
                             '{}_lesion_{}.nii.gz'.format(blob_id, lesion_id))
            )
            nib.save(
                nib.Nifti1Image((markers == lesion_id).astype(np.uint8),
                                img_ref.affine, img_ref.header),
                os.path.join(args.out_lesions_dir,
                             '{}_seed_{}.nii.gz'.format(blob_id, lesion_id))
            )
            n_saved += 1
        logging.info('{} lesion masks + seeds saved to: {}'.format(
            n_saved, args.out_lesions_dir))


if __name__ == '__main__':
    main()
