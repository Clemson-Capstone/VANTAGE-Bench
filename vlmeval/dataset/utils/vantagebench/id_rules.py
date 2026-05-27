"""Deterministic id synthesis rules for the canonical schema layer.

Centralized so that every caller (inference-time emitter, TSV converter,
adapters, tests) derives the same id from the same inputs. Drift between
callers would silently break the submission<->GT join, so all id construction
MUST go through these helpers.
"""


def _video_stem(video):
    """Strip a trailing '.mp4' from a video filename. No other normalization."""
    s = str(video)
    if s.endswith('.mp4'):
        s = s[:-4]
    return s


def make_vqa_id(video, index):
    """Canonical VANTAGE_VQA id.

    Format: f"{video_stem}__q_{index:06d}"

    Verified unique on the live VANTAGE_VQA TSV (1195/1195 rows).
    """
    return f"{_video_stem(video)}__q_{int(index):06d}"


def make_event_verification_id(video, index):
    """Canonical VANTAGE_EventVerification id.

    Format: f"{video_stem}__ev_{index:06d}"

    Verified unique on the live VANTAGE_EventVerification TSV (163/163 rows).
    """
    return f"{_video_stem(video)}__ev_{int(index):06d}"


def make_temporal_id(video, index):
    """Canonical VANTAGE_Temporal id.

    Format: f"{video_stem}__tg_{index:06d}"

    Verified unique on the live VANTAGE_Temporal TSV (1067/1067 rows).
    Note: the Temporal TSV stores video stems without a '.mp4' extension, so
    _video_stem() is effectively a passthrough here; the helper still applies
    its strip rule for safety in case future data carries the suffix.
    """
    return f"{_video_stem(video)}__tg_{int(index):06d}"


def make_dvc_id(video, index):
    """Canonical VANTAGE_DVC id.

    Format: f"{video_stem}__dvc_{index:06d}"

    Verified unique on the live VANTAGE_DVC TSV (104/104 rows). The DVC TSV
    stores video filenames WITH a '.mp4' extension; _video_stem strips it.
    """
    return f"{_video_stem(video)}__dvc_{int(index):06d}"


def _image_stem(image):
    """Strip a trailing image extension from a filename. Case-insensitive.

    Strips: '.jpg', '.jpeg', '.png'. No other normalization.
    Kept distinct from _video_stem so each task's stem rule remains explicit.
    """
    s = str(image)
    lower = s.lower()
    for ext in ('.jpeg', '.jpg', '.png'):
        if lower.endswith(ext):
            return s[: -len(ext)]
    return s


def make_grounding_id(image, index):
    """Canonical VANTAGE_2DGrounding id.

    Format: f"{image_stem}__rx_{index:06d}"

    Verified unique on the live VANTAGE_2DGrounding annotations.json
    (3276/3276 records; multiple referring expressions per image are
    disambiguated by the enumeration index).
    """
    return f"{_image_stem(image)}__rx_{int(index):06d}"


def make_pointing_id(image_path, index):
    """Canonical VANTAGE_2DPointing id.

    Format: f"{image_stem}__sp_{index:06d}"
    where image_stem = _image_stem(os.path.basename(image_path)).

    The 2DPointing TSV stores image_path with a subdirectory prefix
    (e.g. 'images_annotated/000000_000000__largest_in_class_2.jpg'); we use
    basename() to drop the directory before stripping the image extension.

    Verified unique on the live VANTAGE_2DPointing TSV (1005/1005 rows).
    """
    import os
    basename = os.path.basename(str(image_path))
    return f"{_image_stem(basename)}__sp_{int(index):06d}"


def make_astro_id(image_filename, index):
    """Canonical Astro2D id.

    Format: f"{image_stem}__ol_{index:06d}"

    image_filename is expected to be a basename (no subdirectory) — matches
    the Astro2D 'image_filename' column produced by _build_data_structure.

    Verified unique on the live Astro2D dataset (628/628 image files).
    """
    return f"{_image_stem(image_filename)}__ol_{int(index):06d}"


def make_sot_id(seq_dir_name):
    """Canonical VANTAGE_SOT id.

    Format: passthrough of seq_dir.name.

    Unlike prior phase id rules, SOT does NOT add a task token or index
    suffix. The seq_dir basename (e.g.,
    'Warehouse_000__Camera_0003_0005648__obj37') is already a stable
    semantic id — adding a synthetic suffix would be redundant.

    Verified unique on the live VANTAGE_SOT dataset (200/200 sequence dirs).

    The single argument is intentional: SOT id derivation does NOT depend on
    an enumeration index. The whole point of this id is to REPLACE the
    fragile enumeration-int join key used by the legacy evaluator cache.
    """
    return str(seq_dir_name)
