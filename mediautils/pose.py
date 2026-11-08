"""Pictures of a rush and the body keypoints of the people in them, computed once per song.

The preparation of a song's montage (rules agreed with the user, 2026-11-08) starts by
extracting pictures of the rush once, 2 per second, 1920 wide
(:func:`extract_rush_images`); everything after reads them, nothing is computed again
during the back-and-forth with the user.

On these pictures, YOLO pose (:func:`detect_poses`, model ``yolo11m-pose.pt``) gives
17 keypoints per person (:data:`KEYPOINTS`): nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles, each ``[x, y, confidence]``. They are well placed even
when the musicians overlap; but YOLO loses a person standing behind another one. A
**second pass** (:func:`second_pass`) finds them back: the same picture is given again,
cut to the region where they overlap (YOLO then sees them bigger), and mirrored. On
*Sarbacane* (Burlingue, June 2026), the crop alone found back the 65 pictures out of 65
where Mathieu was missing.

The top of the skull is not a keypoint: :func:`head_top` estimates it from the eyes,
the ears and the nose.

:func:`face_direction` measures where a face is turned, from the same keypoints:
the position of the nose between the ears. It is measured relative to the camera; what
it means on stage (who the person looks at) depends on the camera, the person and their
place, and is calibrated from a few moments commented by the user (:class:`GazeScale`).

Who is who is decided by :mod:`mediautils.identify`.

ultralytics (YOLO) is not a dependency of mediautils: run with
``uv run --with ultralytics``. The model file is downloaded by ultralytics the first
time, or given as a path.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""Names of the 17 keypoints of YOLO pose (COCO order); left and right of the person."""

NOSE, EYES, EARS, SHOULDERS, WRISTS, HIPS, ANKLES = (
    0,
    (1, 2),
    (3, 4),
    (5, 6),
    (9, 10),
    (11, 12),
    (15, 16),
)
SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)
"""Pairs of keypoints drawn as the bones of a skeleton."""

SEEN = 0.3
"""Confidence above which a keypoint counts as seen."""


def extract_rush_images(
    video: str | os.PathLike,
    out_dir: str | os.PathLike,
    start: float = 0.0,
    duration: float | None = None,
    fps: float = 2.0,
    width: int = 1920,
    quality: int = 3,
) -> list[Path]:
    """Pictures of a rush, extracted once (requires ffmpeg).

    Picture ``k`` (file ``out_dir/00000.jpg``, ``00001.jpg``...) is the time
    ``start + k / fps`` of the video. If `out_dir` already holds pictures, nothing is
    extracted again: they are returned as they are.

    Parameters
    ----------
    video : str | os.PathLike
        The rush.
    out_dir : str | os.PathLike
        Directory of the pictures (a subdirectory of the song's montage folder: the
        user wants no hidden files elsewhere; erased only at the final clean-up of the
        concert, with the user's approval).
    start : float
        Time of the first picture in the video (seconds), e.g. the start of the song in
        the rush.
    duration : float, optional
        Length to extract (seconds); by default up to the end of the video.
    fps : float
        Pictures per second.
    width : int
        Width of the pictures (pixels); the height keeps the aspect ratio.
    quality : int
        JPEG quality of ffmpeg (2: best, 31: worst).

    Returns
    -------
    list[Path]
        The pictures, in order.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(out.glob("*.jpg"))
    if files:
        return files
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-q:v",
        str(quality),
        "-start_number",
        "0",
        str(out / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out.glob("*.jpg"))


def read_image(path: str | os.PathLike) -> np.ndarray:
    """A picture as a BGR array (OpenCV), from a path that may hold any character."""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not a picture: {path}")
    return img


def _model(model):
    if isinstance(model, (str, os.PathLike)):
        from ultralytics import YOLO

        return YOLO(str(model))
    return model


def _people(result, dx=0.0, dy=0.0):
    """Detections of a YOLO result: ``[{"box": [x0, y0, x1, y1], "kp": [[x, y, c]] * 17}]``,
    shifted by (dx, dy) (a crop back into the coordinates of the picture)."""
    if result.keypoints is None:
        return []
    out = []
    for box, kp in zip(result.boxes.xyxy.tolist(), result.keypoints.data.tolist()):
        out.append(
            {
                "box": [
                    round(box[0] + dx),
                    round(box[1] + dy),
                    round(box[2] + dx),
                    round(box[3] + dy),
                ],
                "kp": [
                    [round(p[0] + dx), round(p[1] + dy), round(p[2], 2)] for p in kp
                ],
            }
        )
    return out


def detect_poses(
    images: list[str | os.PathLike],
    fps: float = 2.0,
    model="yolo11m-pose.pt",
    conf: float = 0.35,
    imgsz: int = 1280,
    min_height: float = 0.0,
    max_top: float | None = None,
    progress: int = 0,
) -> list[dict]:
    """Body keypoints of the people of each picture (YOLO pose).

    Parameters
    ----------
    images : list[str | os.PathLike]
        The pictures (e.g. from :func:`extract_rush_images`); picture ``k`` is the time
        ``k / fps``.
    fps : float
        Pictures per second.
    model : str | os.PathLike | object
        YOLO pose model: a path or name (``"yolo11m-pose.pt"``), or a loaded model (any
        object with the ``predict`` method of ultralytics).
    conf : float
        Least confidence of a detection.
    imgsz : int
        Size at which YOLO looks at the picture (1280: the musicians far from the camera
        are found better than at the default 640).
    min_height : float
        Detections less tall than this are dropped (pixels; e.g. pieces of bodies).
    max_top : float, optional
        Detections whose box starts lower than this are dropped (pixels; e.g. the heads
        of spectators at the bottom of the picture).
    progress : int
        Print the progress every `progress` pictures (0: never).

    Returns
    -------
    list[dict]
        One ``{"t": time, "people": [{"box": [x0, y0, x1, y1], "kp": [[x, y, c]] * 17}]}``
        per picture (keypoints in the order of :data:`KEYPOINTS`).

    Examples
    --------
    With a stand-in for the YOLO model:

        >>> class Result:
        ...     class boxes:
        ...         xyxy = np.array([[100.0, 50, 300, 700], [0, 900, 50, 1000]])
        ...     class keypoints:
        ...         data = np.ones((2, 17, 3))
        >>> class Model:
        ...     def predict(self, img, **kw):
        ...         return [Result()]
        >>> import tempfile, cv2
        >>> d = Path(tempfile.mkdtemp())
        >>> _ = cv2.imwrite(str(d / "00000.jpg"), np.zeros((1080, 1920, 3), np.uint8))
        >>> frames = detect_poses([d / "00000.jpg"], model=Model(), min_height=300)
        >>> frames[0]["t"], frames[0]["people"][0]["box"]
        (0.0, [100, 50, 300, 700])
        >>> len(frames[0]["people"])
        1
    """
    yolo = _model(model)
    frames = []
    for k, f in enumerate(images):
        r = yolo.predict(read_image(f), conf=conf, imgsz=imgsz, verbose=False)[0]
        people = [
            p
            for p in _people(r)
            if p["box"][3] - p["box"][1] >= min_height
            and (max_top is None or p["box"][1] < max_top)
        ]
        frames.append({"t": round(k / fps, 3), "people": people})
        if progress and k % progress == 0:
            print(f"poses : {k} / {len(images)}", flush=True)
    return frames


def same_body(a: list, b: list, tol: float = 60.0) -> bool:
    """Whether two sets of keypoints are the same body: same shoulders, within `tol` pixels.

    Two heads close to each other are not enough: in profile, the nose of someone in
    front passes over the head of the one behind (Mathieu and Andrea, *Sarbacane*).

        >>> a = [[0, 0, 1]] * 5 + [[100, 300, 1], [200, 300, 1]] + [[0, 0, 0]] * 10
        >>> b = [[0, 0, 1]] * 5 + [[120, 310, 1], [210, 290, 1]] + [[0, 0, 0]] * 10
        >>> same_body(a, b), same_body(a, [[0, 0, 1]] * 5 + [[400, 300, 1], [500, 300, 1]] + [[0, 0, 0]] * 10)
        (True, False)
    """
    return (
        abs(a[5][0] - b[5][0]) < tol
        and abs(a[6][0] - b[6][0]) < tol
        and abs(a[5][1] - b[5][1]) < tol
    )


def second_pass(
    frames: list[dict],
    images: list[str | os.PathLike],
    which: list[int],
    region: tuple[float, float, float, float],
    model="yolo11m-pose.pt",
    conf: float = 0.25,
    imgsz: int = 1280,
) -> list[dict]:
    """Second look for the people YOLO lost where two of them overlap.

    The pictures `which` are given again to YOLO cut to `region` (the overlap, with
    some room: the person is then bigger in what YOLO sees), and that crop mirrored too
    (done here: the ``augment`` option of ultralytics is not supported by its pose
    models). The bodies found that are not already in the picture
    (:func:`same_body`) are added, marked ``"second_pass": True``. Who is who is decided
    afterwards (:func:`mediautils.identify.identify`).

    Parameters
    ----------
    frames : list[dict]
        Keypoints of the pictures (:func:`detect_poses`); not modified.
    images : list[str | os.PathLike]
        The pictures.
    which : list[int]
        Indices of the pictures to look at again (those where someone is missing:
        :func:`mediautils.identify.missing`). The second pass is costly; it is run on
        these only.
    region : tuple[float, float, float, float]
        ``(x0, y0, x1, y1)``, pixels of the pictures.
    model, conf, imgsz
        As in :func:`detect_poses`.

    Returns
    -------
    list[dict]
        A copy of `frames` with the bodies found.
    """
    import copy

    yolo = _model(model)
    out = copy.deepcopy(frames)
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    for k in which:
        img = read_image(images[k])[y0:y1, x0:x1]
        w = img.shape[1]
        found = _people(
            yolo.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0], x0, y0
        )
        mirror = yolo.predict(
            img[:, ::-1].copy(), conf=conf, imgsz=imgsz, verbose=False
        )[0]
        found += [_unmirror(p, w, x0) for p in _people(mirror, 0, y0)]
        for p in found:
            if min(p["kp"][5][2], p["kp"][6][2]) < SEEN:
                continue
            if any(same_body(p["kp"], q["kp"]) for q in out[k]["people"]):
                continue
            p["second_pass"] = True
            out[k]["people"].append(p)
    return out


def _unmirror(p, w, dx):
    """A detection in a mirrored crop of width `w`, back into the picture (crop at `dx`):
    x flipped, and the left and right keypoints swapped back.

        >>> kp = [[10 * i, i, 1.0] for i in range(17)]
        >>> q = _unmirror({"box": [10, 0, 50, 100], "kp": kp}, 200, 1000)
        >>> q["box"], q["kp"][0], q["kp"][1], q["kp"][2]
        ([1150, 0, 1190, 100], [1200, 0, 1.0], [1180, 2, 1.0], [1190, 1, 1.0])
    """
    kp = [[dx + w - x, y, c] for x, y, c in p["kp"]]
    for a in range(1, 17, 2):
        kp[a], kp[a + 1] = kp[a + 1], kp[a]
    b = p["box"]
    return {"box": [dx + w - b[2], b[1], dx + w - b[0], b[3]], "kp": kp}


def save_poses(frames: list[dict], path: str | os.PathLike) -> None:
    """Write keypoints (:func:`detect_poses`, :func:`mediautils.identify.identify`) to a JSON file."""
    Path(path).write_text(json.dumps(frames), encoding="utf-8")


def load_poses(path: str | os.PathLike) -> list[dict]:
    """Read keypoints written by :func:`save_poses`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seen(kp: list, i: int) -> bool:
    """Whether keypoint `i` is seen."""
    return kp[i][2] > SEEN


def shoulders_x(kp: list) -> float | None:
    """Middle of the shoulders, else the nose, else None.

    The shoulders place someone better than their head, which leaves their place when
    they lean (Fanfan).
    """
    if seen(kp, 5) and seen(kp, 6):
        return (kp[5][0] + kp[6][0]) / 2
    if seen(kp, NOSE):
        return float(kp[NOSE][0])
    return None


def head_scale(kp: list) -> float:
    """Half the width of the head, in pixels: a quarter of the shoulders' width (bounded),
    else the span of the ears, else 35."""
    if seen(kp, 5) and seen(kp, 6):
        return float(np.clip(abs(kp[5][0] - kp[6][0]) * 0.25, 22, 60))
    if seen(kp, 3) and seen(kp, 4):
        return float(np.clip(abs(kp[3][0] - kp[4][0]) / 2, 15, 60))
    return 35.0


def eyes_y(kp: list) -> float | None:
    """Height of the eyes, else a little above the nose, else None."""
    ys = [kp[i][1] for i in EYES if seen(kp, i)]
    if ys:
        return float(np.mean(ys))
    if seen(kp, NOSE):
        return float(kp[NOSE][1]) - 10
    return None


def head_top(kp: list) -> float | None:
    """Estimated top of the skull (pixels), from the eyes and the size of the head.

    >>> head = [[500, 300, 0.9], [490, 290, 0.9], [510, 290, 0.9], [0, 0, 0], [0, 0, 0]]
    >>> kp = head + [[400, 400, 0.9], [600, 400, 0.9]] + [[0, 0, 0]] * 10
    >>> head_top(kp)
    215.0
    """
    e = eyes_y(kp)
    return None if e is None else e - 1.5 * head_scale(kp)


def head_x(kp: list) -> float | None:
    """Horizontal centre of the head: between the nose and the visible ears (in profile,
    the nose is at the edge of the head)."""
    if not seen(kp, NOSE):
        return None
    ears = [kp[i][0] for i in EARS if seen(kp, i)]
    return (kp[NOSE][0] + float(np.mean(ears))) / 2 if ears else float(kp[NOSE][0])


def body_box(kp: list) -> tuple[float, float, float, float] | None:
    """``(x0, top, x1, bottom)`` of a body from its keypoints: from the top of the skull
    to the lowest seen point, between the extreme seen points (without the instrument)."""
    pts = [p for p in kp if p[2] > SEEN]
    if not pts:
        return None
    top = head_top(kp)
    ys = [p[1] for p in pts]
    return (
        float(min(p[0] for p in pts)),
        float(min(ys) if top is None else min(top, min(ys))),
        float(max(p[0] for p in pts)),
        float(max(ys)),
    )


def face_direction(kp: list, min_conf: float = SEEN) -> float | None:
    """Where a face is turned, relative to the camera: position of the nose between the
    ears, ``(nose_x - middle of the ears) / span of the ears``.

    Negative: towards the left of the picture; about 0: facing the camera. The ears are
    taken even when YOLO guesses a hidden one: beyond -1 (or +1), the values (-3, -9)
    only mean "frankly". None if the nose is not seen.

    What it means on stage depends on the camera (a side camera: facing it is not facing
    the audience) and on the person: see :class:`GazeScale`.

        >>> kp = [[480, 300, 0.9]] + [[0, 0, 0]] * 2 + [[540, 300, 0.4], [460, 300, 0.2]] + [[0, 0, 0]] * 12
        >>> face_direction(kp)
        -0.25
        >>> face_direction([[480, 300, 0.1]] + kp[1:]) is None
        True
    """
    nose, left, right = kp[NOSE], kp[3], kp[4]
    if nose[2] < min_conf:
        return None
    mid = (left[0] + right[0]) / 2
    return float((nose[0] - mid) / max(abs(left[0] - right[0]), 5))


@dataclass
class GazeScale:
    """What a face direction (:func:`face_direction`) means, for a camera, a person and
    their place, calibrated from a few moments commented by the user.

    Two frames of reference: the **measure**, relative to the camera, decides the margin
    in the frame; the **word**, relative to the stage, says whom or what the person looks
    at, hence whom to show.

    Attributes
    ----------
    bounds : list[tuple[float, str]]
        ``(upper bound, word)``, increasing bounds; the last word holds above the last
        bound.
    last : str
        Word above the last bound.

    Examples
    --------
    Fanfan on the side camera of Burlingue (Canon R7, court side, June 2026), calibrated
    on *Sarbacane* (below -1: Thomas; about -0.35: his music stand; about 0: facing the
    camera, i.e. rather right on stage; about +0.85: Mathieu; the limit between
    "rather" and "frankly" right is a guess):

        >>> fanfan = GazeScale([(-0.7, "Thomas"), (-0.15, "pupitre"), (0.45, "côté cour")], "Mathieu")
        >>> [fanfan.word(v) for v in (-3.0, -0.35, 0.02, 0.85)]
        ['Thomas', 'pupitre', 'côté cour', 'Mathieu']
        >>> fanfan.word(None) is None
        True
    """

    bounds: list[tuple[float, str]]
    last: str

    def word(self, value: float | None) -> str | None:
        """Word for a measure (None for None)."""
        if value is None:
            return None
        for bound, word in self.bounds:
            if value < bound:
                return word
        return self.last
