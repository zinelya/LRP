import logging
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.feature import peak_local_max
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

#SOURCE: scilpy/image/labels.py
def get_data_as_labels(in_img):
    """
    Get data as label (force type np.uint16), check data type before casting.

    Parameters
    ----------
    in_img: nibabel.nifti1.Nifti1Image
        Image.

    Return
    ------
    data: numpy.ndarray
        Data (dtype: np.uint16).
    """
    curr_type = in_img.get_data_dtype()

    if np.issubdtype(curr_type, np.signedinteger) or \
            np.issubdtype(curr_type, np.unsignedinteger):
        return np.asanyarray(in_img.dataobj).astype(np.uint16)
    else:
        basename = os.path.basename(in_img.get_filename())
        raise IOError('The image {} cannot be loaded as label because '
                      'its format {} is not compatible with a label '
                      'image'.format(basename, curr_type))


def get_labels_from_mask(mask_data, labels=None, background_label=0,
                         min_voxel_count=0):
    """
    Get labels from a binary mask which contains multiple blobs. Each blob
    will be assigned a label, by default starting from 1. Background will
    be assigned the background_label value.

    Parameters
    ----------
    mask_data: np.ndarray
        The mask data.
    labels: list, optional
        Labels to assign to each blobs in the mask. Excludes the background
        label.
    background_label: int
        Label for the background.
    min_voxel_count: int, optional
        Minimum number of voxels for a blob to be considered. Blobs with fewer
        voxels will be ignored.

    Returns
    -------
    label_map: np.ndarray
        The labels.
    """
    # Get the number of structures and assign labels to each blob
    label_map, nb_structures = ndi.label(mask_data)
    if min_voxel_count:
        new_count = 0
        for label in range(1, nb_structures + 1):
            if np.count_nonzero(label_map == label) < min_voxel_count:
                label_map[label_map == label] = 0
            else:
                new_count += 1
                label_map[label_map == label] = new_count
        logging.debug(
            f"Ignored blob {nb_structures-new_count} with fewer "
            "than {min_voxel_count} voxels")
        nb_structures = new_count

    # Assign labels to each blob if provided
    if labels:
        # Only keep the first nb_structures labels if the number of labels
        # provided is greater than the number of blobs in the mask.
        if len(labels) > nb_structures:
            logging.warning("Number of labels ({}) does not match the number "
                            "of blobs in the mask ({}). Only the first {} "
                            "labels will be used.".format(
                                len(labels), nb_structures, nb_structures))
        # Cannot assign fewer labels than the number of blobs in the mask.
        elif len(labels) < nb_structures:
            raise ValueError("Number of labels ({}) is less than the number of"
                             " blobs in the mask ({}).".format(
                                 len(labels), nb_structures))

        # Copy the label map to avoid scenarios where the label list contains
        # labels that are already present in the label map
        custom_label_map = label_map.copy()
        # Assign labels to each blob
        for idx, label in enumerate(labels[:nb_structures]):
            custom_label_map[label_map == idx + 1] = label
        label_map = custom_label_map

    logging.info('Assigned labels {} to the mask.'.format(
        np.unique(label_map[label_map != background_label])))

    if background_label != 0 and background_label in label_map:
        logging.warning("Background label {} corresponds to a label "
                        "already in the map. This will cause issues.".format(
                            background_label))

    # Assign background label
    if background_label:
        label_map[label_map == 0] = background_label

    return label_map


def _norm(x):
    """Normalise array to [0, 1]; returns zeros if constant."""
    if x.max() == x.min():
        return x * 0
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _geodesic_distance_transform(mask, image, lambda_=1.0):
    """
    Compute a geodesic distance transform from the mask boundary inward.

    Each voxel's distance is the minimum cumulative cost of any path from
    the mask boundary to that voxel, where the local cost is:

        cost(x) = 1 + lambda_ * norm(||∇image(x)||)

    High-gradient regions (e.g. the iron rim in QSM) act as barriers,
    making the geodesic distance to the centre larger than the Euclidean
    distance whenever the path must cross an intensity edge.

    Parameters
    ----------
    mask : numpy.ndarray (bool)
        Binary lesion mask.
    image : numpy.ndarray
        Image used to compute gradient magnitude (e.g. QSM, FLAIR).
    lambda_ : float
        Weight of the gradient term in the cost function.

    Returns
    -------
    geodesic_dist : numpy.ndarray
        Geodesic distance from the boundary, zero outside the mask.
    """
    grad_mag = np.sqrt(sum(g ** 2 for g in np.gradient(image)))

    cost = np.full(mask.shape, 1e10)
    cost[mask] = 1.0 + lambda_ * _norm(grad_mag)[mask]

    boundary = mask & ~ndi.binary_erosion(mask)
    boundary_coords = list(zip(*np.where(boundary)))

    geodesic_dist, _ = MCP_Geometric(cost).find_costs(boundary_coords)
    geodesic_dist[~mask] = 0.0

    return geodesic_dist


def _anisotropic_diffusion_distance(mask, image, lambda_=1.0):
    """
    Compute an anisotropic diffusion distance from the mask boundary inward.

    Unlike the geodesic distance (where cost depends only on location), the
    cost here depends on both location AND direction of movement:

        w(i→j) = ||d|| * (1 + lambda_ * cos²(d̂, ∇I_mid))

    where d̂ is the unit step direction and ∇I_mid is the normalised gradient
    averaged at the two endpoints. This means:
    - Moving along an intensity edge (d ⊥ ∇I): cos = 0, cost = ||d|| (cheap)
    - Moving across an intensity edge (d ∥ ∇I): cos = 1, cost = ||d||*(1+λ) (expensive)

    The resulting distance is large wherever the path from the boundary must
    cross a strong gradient (e.g. the iron rim in a PRL) regardless of
    Euclidean length, and explicitly allows paths that travel along the rim
    without penalty.

    Uses a 6-connectivity sparse graph with scipy Dijkstra (multi-source from
    all boundary voxels).

    Parameters
    ----------
    mask : numpy.ndarray (bool)
        Binary lesion mask.
    image : numpy.ndarray
        Image used to compute gradient direction (e.g. QSM, FLAIR).
    lambda_ : float
        Anisotropy strength: 0 = isotropic (reduces to step length),
        large = strongly penalises crossing gradients.

    Returns
    -------
    aniso_dist : numpy.ndarray
        Anisotropic distance from the boundary, zero outside the mask.
    """
    shape = mask.shape
    n = mask.size

    grads = np.gradient(image)
    grad_vec = np.stack(grads, axis=-1)                    # (*shape, 3)
    grad_mag = np.sqrt(np.sum(grad_vec ** 2, axis=-1) + 1e-8)
    grad_norm = grad_vec / grad_mag[..., np.newaxis]       # unit gradient

    flat_idx = np.arange(n).reshape(shape)

    # 3 positive offsets — edges are added symmetrically
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    all_rows, all_cols, all_w = [], [], []

    for off in offsets:
        step_len = float(np.linalg.norm(off))
        d_hat = np.array(off, dtype=float) / step_len

        # Source and target slices
        src_sl = tuple(slice(None, -o if o else None) for o in off)
        tgt_sl = tuple(slice(o if o else None, None) for o in off)

        valid = mask[src_sl] & mask[tgt_sl]
        if not valid.any():
            continue

        # cos²(step, gradient) averaged at both endpoints
        cos_src = np.sum(d_hat * grad_norm[src_sl], axis=-1)
        cos_tgt = np.sum(d_hat * grad_norm[tgt_sl], axis=-1)
        cos_sq = 0.5 * (cos_src ** 2 + cos_tgt ** 2)

        w = step_len * (1.0 + lambda_ * cos_sq)

        i_flat = flat_idx[src_sl][valid]
        j_flat = flat_idx[tgt_sl][valid]
        w_valid = w[valid]

        # Symmetric edges
        all_rows.append(np.concatenate([i_flat, j_flat]))
        all_cols.append(np.concatenate([j_flat, i_flat]))
        all_w.append(np.concatenate([w_valid, w_valid]))

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    weights = np.concatenate(all_w)

    graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

    boundary = mask & ~ndibinary_erosion(mask)
    sources = np.where(boundary.ravel())[0]

    dist = dijkstra(graph, indices=sources, min_only=True)

    aniso_dist = dist.reshape(shape)
    aniso_dist[~mask] = 0.0

    return aniso_dist


def watershed_lesions(mask, alpha=0.7, beta=0.3, min_distance=3,
                      w_flair=1.0, w_t1=1.5,
                      distance_mode='euclidean', geodesic_lambda=1.0,
                      blend=0.0, seed_mode='seed_map',
                      flair=None, t1=None, qsm=None):
    """
    Split a binary lesion mask into individual labels using watershed.

    Builds a seed map and an energy (relief) map from optional multimodal
    arrays (FLAIR, T1, QSM) and a distance transform, then runs watershed.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary lesion mask (bool or 0/1).
    alpha : float
        Weight of QSM contribution to the energy map.
    beta : float
        Weight of the distance transform subtracted from the energy map.
    min_distance : int
        Minimum distance (voxels) between detected seed peaks.
    w_flair : float
        Weight of FLAIR contribution to the seed map.
    w_t1 : float
        Weight of T1 contribution to the seed map.
    distance_mode : str
        Distance transform used to locate lesion centres and modulate the
        energy map. One of 'euclidean' (default), 'geodesic', or
        'anisotropic'.
        - 'euclidean'   : standard EDT, location-only cost.
        - 'geodesic'    : isotropic gradient-weighted shortest path; cost at
                          each voxel = 1 + λ·||∇I||. Barriers are equally
                          expensive in all directions.
        - 'anisotropic' : direction-dependent cost; crossing a gradient is
                          expensive (cos²(d̂,∇I) penalty) but moving along
                          the gradient is cheap. Paths can freely travel the
                          rim without accumulating cost.
        Image priority for non-euclidean modes: QSM > FLAIR > T1.
    geodesic_lambda : float
        Strength of the gradient/directional penalty used in 'geodesic' and
        'anisotropic' modes.
    blend : float
        Mixing weight between the Euclidean distance and the selected
        non-Euclidean distance (0 = pure Euclidean, 1 = pure non-Euclidean,
        0.5 = equal mix). Ignored when distance_mode='euclidean'.
        Both maps are normalised to [0,1] before blending so their scales
        are comparable.
    seed_mode : str
        Strategy used to detect seed positions. One of:
        - 'seed_map' (default): composite map driven by FLAIR/T1 brightness
          and QSM rim penalty; peaks = lesion centres.
        - 'energy'  : seeds are local maxima of (1 - energy_map), i.e.
          valleys of the energy surface; no separate seed map is built.
    flair : numpy.ndarray, optional
        FLAIR volume; bright signal drives seed detection.
    t1 : numpy.ndarray, optional
        T1 volume; dark signal (black holes) drives seed detection.
    qsm : numpy.ndarray, optional
        QSM volume; positive iron drives the energy map.

    Returns
    -------
    labels : numpy.ndarray (int32)
        Label image with one unique integer per separated lesion.
    markers : numpy.ndarray (int32)
        Seed voxel image (one voxel per detected centre).
    """

    seed_map = np.zeros(mask.shape)
    energy_map = np.zeros(mask.shape)

    if flair is not None:
        f_n = _norm(flair * mask)
        seed_map += w_flair * f_n
        energy_map += 1.0 - f_n

    if t1 is not None:
        t_n = _norm(t1 * mask)
        seed_map += w_t1 * (1.0 - t_n)
        energy_map += t_n

    edt_map = ndi.distance_transform_edt(mask)

    if distance_mode in ('geodesic', 'anisotropic'):
        ref_image = qsm if qsm is not None else (t1 if t1 is not None else flair)
        if ref_image is None:
            logging.warning('No image available for %s distance, '
                            'falling back to euclidean.', distance_mode)
            dist_map = edt_map
        else:
            if distance_mode == 'geodesic':
                noneuclid_map = _geodesic_distance_transform(mask, ref_image,
                                                             geodesic_lambda)
            else:
                noneuclid_map = _anisotropic_diffusion_distance(mask, ref_image,
                                                                geodesic_lambda)
            dist_map = (1.0 - blend) * _norm(edt_map) + blend * _norm(noneuclid_map)
    else:
        dist_map = edt_map

    dt_n = _norm(dist_map)

    if qsm is not None:
        q_pos = np.maximum(0, qsm) * mask
        energy_map += alpha * _norm(q_pos)
        rim_mask = q_pos * (1.0 - dt_n)
        seed_map -= 0.8 * _norm(rim_mask)
        seed_map += 0.2 * _norm(q_pos * dt_n)

    energy_map = _norm(energy_map - beta * _norm(dist_map))
    energy_map += np.random.normal(0, 1e-6, energy_map.shape)

    if seed_mode == 'energy':
        source = 1.0 - energy_map
    else:
        source = seed_map

    coords = peak_local_max(source, min_distance=min_distance, labels=mask)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, coord in enumerate(coords):
        markers[tuple(coord)] = i + 1

    if len(coords) <= 1:
        # Single seed (or none): no need to flood — the whole blob is one lesion.
        labels = mask.astype(np.int32)
        return labels, markers

    labels = watershed(energy_map, markers, mask=mask)

    return labels.astype(np.int32), markers

def absorb_small_confluant_labels(labels, min_voxels=10):
    """
    Remove small confluant labels and reassign their voxels to the nearest kept label.

    Labels with fewer than `min_voxels` voxels are absorbed into whichever
    kept label is closest (Euclidean distance transform on the kept-label mask).

    Parameters
    ----------
    labels : numpy.ndarray (int)
        3D watershed label map.
    min_voxels : int
        Labels with fewer voxels than this are absorbed.

    Returns
    -------
    cleaned : numpy.ndarray (int)
        Label map with small labels reassigned.
    """
    cleaned = labels.copy()
    unique, counts = np.unique(labels[labels > 0], return_counts=True)

    small = unique[counts < min_voxels]
    keep  = unique[counts >= min_voxels]

    if len(small) == 0:
        return cleaned

    keep_mask   = np.isin(cleaned, keep)
    kept_labels = np.where(keep_mask, cleaned, 0)

    # Compute EDT once — nearest_idx[axis, i, j, k] gives the coordinate of
    # the closest kept-label voxel from position (i, j, k)
    _, nearest_idx = ndi.distance_transform_edt(~keep_mask, return_indices=True)

    small_region = np.isin(labels, small)
    ni = nearest_idx[:, small_region]
    cleaned[small_region] = kept_labels[ni[0], ni[1], ni[2]]

    return cleaned