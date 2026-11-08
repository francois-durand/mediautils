"""Pictures of a rush, keypoints, who is who, catalogue of framings, alarms."""

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from mediautils import (
    Alarm,
    Extent,
    GazeScale,
    Person,
    Thresholds,
    Zone,
    alarm_log,
    catalogue,
    catalogue_alarms,
    detect_poses,
    draw_frame,
    extract_rush_images,
    face_direction,
    identify,
    intended,
    load_poses,
    missing,
    save_poses,
    second_pass,
    shot_alarms,
    skull_lightness,
    usual_places,
    variant_name,
)
from mediautils.alarms import clock, plan
from mediautils.identify import gaze_series, swap_names, who
from mediautils.pose import _unmirror, body_box, head_top, read_image


def body(x, y=300, w=160, torso=300, feet=None, wrists=None, nose=0.9):
    """Keypoints of a person facing the camera, nose at (x, y)."""
    kp = [[x, y, nose], [x - 10, y - 10, 0.9], [x + 10, y - 10, 0.9]]
    kp += [[x + 30, y, 0.9], [x - 30, y, 0.9]]
    kp += [[x - w / 2, y + 100, 0.9], [x + w / 2, y + 100, 0.9]]
    kp += [[0, 0, 0]] * 2
    wy = y + 300 if wrists is None else wrists
    kp += [[x - w / 2 - 10, wy, 0.9], [x + w / 2 + 10, wy, 0.9]]
    kp += [[x - w / 3, y + 100 + torso, 0.9], [x + w / 3, y + 100 + torso, 0.9]]
    kp += [[0, 0, 0]] * 2
    kp += [[x, feet, 0.9]] * 2 if feet else [[0, 0, 0]] * 2
    return {"box": [x - w, y - 80, x + w, feet or y + 500], "kp": kp}


class Model:
    """Stand-in for a YOLO model: returns the given detections, one list per call (the
    last one again after)."""

    def __init__(self, *responses):
        self.responses, self.calls = responses, []

    def predict(self, img, **kw):
        self.calls.append((img.shape, kw))
        people = self.responses[min(len(self.calls), len(self.responses)) - 1]

        class R:
            class boxes:
                xyxy = np.array([p["box"] for p in people], dtype=float).reshape(-1, 4)

            class keypoints:
                data = np.array([p["kp"] for p in people], dtype=float).reshape(
                    -1, 17, 3
                )

        if not people:
            R.keypoints = None
        return [R()]


@pytest.fixture
def pictures(tmp_path):
    files = []
    for k in range(3):
        f = tmp_path / "img" / f"{k:05d}.jpg"
        f.parent.mkdir(exist_ok=True)
        img = np.full((1080, 1920, 3), 40, np.uint8)
        cv2.imwrite(str(f), img)
        files.append(f)
    return files


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed")
def test_extract_rush_images(tmp_path):
    video = tmp_path / "rush.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=320x180:d=3:r=25",
            str(video),
        ],
        check=True,
    )
    files = extract_rush_images(
        video, tmp_path / "images", start=0.5, duration=2, width=160
    )
    assert len(files) in (4, 5)
    assert read_image(files[0]).shape[:2] == (90, 160)
    # once extracted, not again
    files[-1].unlink()
    assert len(extract_rush_images(video, tmp_path / "images")) == len(files) - 1


def test_read_image_not_a_picture(tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"nothing")
    with pytest.raises(ValueError):
        read_image(f)


def test_detect_and_second_pass(pictures, tmp_path):
    a, b = body(500, wrists=600), body(1500, wrists=600)
    frames = detect_poses(pictures, model=Model([a]), max_top=700, progress=2)
    assert [f["t"] for f in frames] == [0.0, 0.5, 1.0]
    assert all(len(f["people"]) == 1 for f in frames)
    empty = detect_poses(pictures[:1], model=Model([]))
    assert empty[0]["people"] == []
    # the second pass finds b in a crop of the picture; a, already there, is not added twice
    crop_b = {
        "box": [v - 1100 if i % 2 == 0 else v - 200 for i, v in enumerate(b["box"])],
        "kp": [[x - 1100, y - 200, c] for x, y, c in b["kp"]],
    }
    crop_a = {"box": a["box"], "kp": [[x - 1100, y - 200, c] for x, y, c in a["kp"]]}
    model = Model([crop_a, crop_b], [])
    more = second_pass(frames, pictures, [1], (1100, 200, 1920, 1080), model=model)
    assert len(frames[1]["people"]) == 1  # not modified
    assert len(more[1]["people"]) == 2 and more[1]["people"][1]["second_pass"]
    assert more[1]["people"][1]["kp"][5][0] == b["kp"][5][0]
    assert [c[0][:2] for c in model.calls] == [
        (880, 820),
        (880, 820),
    ]  # the crop, then mirrored
    save_poses(more, tmp_path / "pose.json")
    assert load_poses(tmp_path / "pose.json") == more
    # found in the mirrored crop only: put back in place, left and right swapped back
    mirrored = _unmirror(crop_b, 820, 0)
    more = second_pass(
        frames, pictures, [1], (1100, 200, 1920, 1080), model=Model([], [mirrored])
    )
    found = more[1]["people"][1]
    assert found["kp"][5][:2] == b["kp"][5][:2] and found["kp"][6][:2] == b["kp"][6][:2]
    # only the bodies looked for
    model = Model([crop_a, crop_b], [])
    kept = second_pass(
        frames,
        pictures,
        [1],
        (1100, 200, 1920, 1080),
        model=model,
        keep=lambda p: p["kp"][0][0] < 1000,
    )
    assert len(kept[1]["people"]) == 1


def test_keypoint_helpers():
    p = body(500, nose=0.1)
    assert face_direction(p["kp"]) is None
    assert head_top(p["kp"]) is not None  # from the eyes
    blind = [[0, 0, 0]] * 17
    assert head_top(blind) is None and body_box(blind) is None
    scale = GazeScale([(-0.5, "gauche")], "droite")
    assert scale.word(-1) == "gauche" and scale.word(0) == "droite"


def frames_of(*rows):
    return [{"t": k / 2, "people": people} for k, people in enumerate(rows)]


def test_identify_zones_and_continuity():
    zones = [Zone(("T",), 0, 800), Zone(("A", "M"), 1200, front="M")]
    # A and M side by side, A on the left; on one picture, the one on the left looks in front
    rows = []
    for k in range(7):
        if k == 3:
            rows.append(
                [
                    body(400),
                    body(1300, torso=400, feet=1000),
                    body(1500, torso=300, feet=950),
                ]
            )
        else:
            rows.append([body(400), body(1300), body(1500)])
    named, flags = identify(frames_of(*rows), zones)
    # without a sign nor a clear front, the order left-right is supposed
    assert all(
        f.message.startswith("aucun indice")
        for f in flags
        if "inversion" not in f.message
    )
    assert [g.index for g in flags if "inversion" in g.message] == [3]
    assert who(named[3], "A")["kp"][0][0] == 1300  # repaired: a name follows its person
    assert missing(named, "T") == []
    swap_names(named, 3, "A", "M")
    assert who(named[3], "A")["kp"][0][0] == 1500
    assert len(gaze_series(named, "T")) == 7
    with pytest.raises(ValueError):
        Zone(("A", "B", "C"))


def test_identify_with_a_sign(pictures):
    # a light skull (bald) and a dark one (cap)
    img = np.full((1080, 1920, 3), 40, np.uint8)
    img[180:290, 1450:1550] = 250  # skull of the person at x = 1500
    cv2.imwrite(str(pictures[0]), img)
    zones = [
        Zone(
            ("Andrea", "Mathieu"),
            1200,
            sign=skull_lightness,
            threshold=134,
            front="Andrea",
        )
    ]
    big, small = body(1300, torso=400, feet=1000), body(1500, torso=300, feet=950)
    rows = [[big, small], [body(1500)], [body(1300), body(1500)]]
    named, flags = identify(frames_of(*rows), zones, pictures)
    assert who(named[0], "Mathieu")["kp"][0][0] == 1500
    assert who(named[0], "Mathieu")["sign"] > 134
    # the sign and the front agree (Andrea, dark, is the big one): nothing to look at
    assert not any(g.index == 0 for g in flags)
    # one person alone, told by the threshold
    assert who(named[1], "Andrea") is not None  # dark picture: below the threshold
    # both on the same side of the threshold: flagged
    assert any(g.index == 2 and "ambigu" in g.message for g in flags)
    with pytest.raises(ValueError):
        identify(frames_of(*rows), zones)  # a sign needs the pictures


def test_identify_cases():
    zones = [Zone(("A", "M"), 1200, front="M")]
    # one person alone in a zone of two, without a sign: not named
    named, flags = identify(frames_of([body(1300)]), zones)
    assert who(named[0], "A") is None and "seule personne" in flags[0].message
    # the front decides
    named, _ = identify(
        frames_of([body(1300, torso=400, feet=1000), body(1500, torso=300, feet=950)]),
        zones,
    )
    assert who(named[0], "M")["kp"][0][0] == 1300
    # a duplicate skeleton on the same body is dropped; a body without head is not named
    dup = body(400)
    named, _ = identify(
        frames_of([dup, dict(dup), body(400, y=800)]),
        [Zone(("T",), 0, 800)],
        max_eyes=650,
    )
    assert (
        len(named[0]["people"]) == 2
        and sum("name" in p for p in named[0]["people"]) == 1
    )


def test_identify_one_sign_known(pictures):
    img = np.full((1080, 1920, 3), 40, np.uint8)
    cv2.imwrite(str(pictures[0]), img)
    zones = [
        Zone(
            ("Andrea", "Mathieu"),
            1200,
            sign=skull_lightness,
            threshold=134,
            front="Mathieu",
        )
    ]
    # the back of M is turned: no eyes, no nose; A, in front, is dark (below the threshold)
    m = body(1500, torso=300, feet=950)
    for i in range(5):
        m["kp"][i][2] = 0.0
    named, flags = identify(
        frames_of([body(1300, torso=400, feet=1000), m]), zones, pictures
    )
    assert who(named[0], "Andrea")["kp"][0][0] == 1300
    assert who(named[0], "Mathieu") is not None
    assert any("derrière" in g.message for g in flags)


def places_of(frames, **kw):
    return usual_places(frames, ["T", "F", "L"], **kw)


def test_usual_places_and_catalogue():
    rows = [
        [
            {**body(300), "name": "T"},
            {**body(900), "name": "F"},
            {**body(1500), "name": "L"},
        ]
        for _ in range(6)
    ]
    rows.append(
        [{**body(300), "name": "T"}, {**body(1200), "name": "F"}]
    )  # F away once
    frames = frames_of(*rows)
    places = places_of(
        frames,
        extend={
            "T": Extent(left=80),
            "L": Extent(alone=100),
            "F": Extent(tight_hands=False),
        },
    )
    assert list(places) == ["T", "F", "L"]
    assert places["F"].x0 == 900 - 90  # the usual place, not the mean
    assert places["L"].alone == places["L"].bottom + 100
    assert (
        places_of(frames, since=3.0)["F"].x0 == 1200 - 90
    )  # a variant for the end of the song
    assert places_of(frames, until=0.1).keys() == {"T", "F", "L"}
    z = catalogue(
        places, camera="R7", fixed={"F @0:12,0": (900, 500, 2.0), "F": (905, 500, 2.0)}
    )
    assert len(z) == 3 * 3 + 3 + 1  # 6 intervals, 2 tight shots each, + a variant
    assert z["F"] == ("R7", 905.0, 500.0, 2.0)
    assert list(z)[0] == "large" and list(z)[-1] == "F @0:12,0"
    assert intended("T-L", list(places)) == ["T", "F", "L"]
    assert variant_name("F (fin)", 12) == "F @0:12,0"
    one = catalogue({"T": places["T"]})
    assert "T" in one and "large" not in one


def test_shot_alarms():
    order = ["T", "F", "L"]
    rows = []
    for k in range(20):
        f = body(1000)
        if (
            k >= 10
        ):  # F turned to the left, at the left edge of the frame of the second shot
            f["kp"][0][0] -= 25
        rows.append(
            [
                {**body(300), "name": "T"},
                {**f, "name": "F"},
                {**body(1600), "name": "L"},
                body(1200, y=700),
            ]
        )  # someone not identified
    frames = frames_of(*rows)
    framings = {
        "T": ("C", 300, 500, 2.5),
        "F-L": ("C", 1385, 560, 1.6),
        "L": ("C", 1600, 520, 2.6),
        "L serré": ("C", 1600, 520, 2.7),
        "large": ("C", 960, 540, 1.0),
        "large2": ("C", 960, 540, 1.05),
    }
    shots = plan([(0, "T"), (5, "F-L"), (8, "L"), (8.5, "L serré"), (9, "large")], 10)
    alarms = shot_alarms(
        shots,
        framings,
        frames,
        order,
        singer="F",
        sings=lambda a, b: 1.0 if b <= 5 else 0.0,
        guitarists=("T",),
        fixed={"toms": ("L", (1400, 600, 1700, 800))},
        accents=[8.5],
        attacks=[0.0, 5.0, 8.5],
        offset=0.0,
    )
    kinds = {(a.kind, a.message.split(" (")[0]) for a in alarms}
    assert ("chanteur", "F pas à l'image alors qu'il chante") in kinds
    assert ("musicien", "quelqu'un de non identifié dans le cadre") in kinds
    assert ("enchaînement", "deux cadrages trop proches") in kinds
    assert ("coupe", "coupe loin de toute attaque") in kinds
    assert (
        "coupe",
        "coupe au milieu d'une phrase, sans respiration ni accent",
    ) not in kinds  # 5: F stops singing
    assert all(isinstance(a, Alarm) for a in alarms)
    # going back to a wider framing of the same people
    back = shot_alarms(
        plan([(0, "L serré"), (2, "L")], 4),
        {"L": ("C", 1600, 520, 1.5), "L serré": ("C", 1600, 520, 2.6)},
        frames,
        order,
    )
    assert any("recul" in a.message for a in back)
    # a wide shot with too much margin, moved musicians, hands out
    rows = [[{**body(700), "name": "T"}, {**body(1000), "name": "F"}] for _ in range(4)]
    w = shot_alarms(
        plan([(0, "large")], 2),
        {"large": ("C", 960, 540, 1.0)},
        frames_of(*rows),
        ["T", "F"],
    )
    assert any(a.message.startswith("vide") for a in w)
    hands = shot_alarms(
        plan([(0, "T")], 2),
        {"T": ("C", 700, 400, 3.5)},
        frames_of(*rows),
        ["T", "F"],
        guitarists=("T",),
    )
    assert any("mains de T" in a.message for a in hands)
    moving = frames_of(
        *([[{**body(700), "name": "T"}]] * 8 + [[{**body(1100), "name": "T"}]] * 2)
    )
    moved = shot_alarms([(4.0, 5.0, "T")], {"T": ("C", 900, 500, 1.5)}, moving, ["T"])
    assert any("déplacé" in a.message for a in moved)
    # in a variant of the catalogue for that part (they stand elsewhere there), not moved
    framings = {"T": ("C", 900, 500, 1.5), "T (fin)": ("C", 900, 500, 1.5)}
    moved = shot_alarms(
        [(4.0, 5.0, "T (fin)")], framings, moving, ["T"], variants={"fin": [(4.0, 5.0)]}
    )
    assert not any("déplacé" in a.message for a in moved)


def test_shot_alarms_singer_at_the_edge():
    f = body(1000)
    f["kp"][0][0] = 975  # turned to the left of the picture
    frames = frames_of(*([[{**f, "name": "F"}]] * 4))
    alarms = shot_alarms(
        plan([(0, "F")], 2),
        {"F": ("C", 1140, 500, 4.0)},
        frames,
        ["F"],
        singer="F",
        sings=lambda a, b: 1.0,
        thresholds=Thresholds(singer_edge=0.1, alone_offset=1.0),
    )
    assert any("bord gauche" in a.message for a in alarms)
    g = body(1000)
    g["kp"][0][0] = 1025
    frames = frames_of(*([[{**g, "name": "F"}]] * 4))
    alarms = shot_alarms(
        plan([(0, "F")], 2),
        {"F": ("C", 860, 500, 4.0)},
        frames,
        ["F"],
        singer="F",
        sings=lambda a, b: 1.0,
        thresholds=Thresholds(singer_edge=0.1, alone_offset=1.0),
    )
    assert any("bord droit" in a.message for a in alarms)


def test_cut_alarms_and_headless_body():
    frames = frames_of(*([[{**body(1000), "name": "F"}]] * 10))
    shots = plan([(0, "a"), (2.5, "b")], 5)
    alarms = shot_alarms(
        shots,
        {},
        frames,
        ["F"],
        singer="F",
        sings=lambda a, b: 1.0,
        accents=[],
        attacks=[],
    )
    msgs = [a.message for a in alarms]
    assert "coupe au milieu d'une phrase, sans respiration ni accent" in msgs
    assert any(m.startswith("coupe loin") for m in msgs)
    # the frame cuts the head off
    alarms = shot_alarms(
        plan([(0, "F bas")], 4), {"F bas": ("C", 1000, 530, 3.0)}, frames, ["F"]
    )
    assert any("sans sa tête" in a.message for a in alarms)
    alarms = shot_alarms(
        plan([(0, "F")], 4), {"F": ("C", 1000, 560, 2.0)}, frames, ["F"]
    )
    assert any("coupée par le bord haut" in a.message for a in alarms)


def test_catalogue_alarms_heads():
    places = {
        "A": Person(300, 700, 100, 150, 600),
        "B": Person(1000, 1700, 400, 450, 800),
    }
    z = {"A": ("L", 500, 300, 2.5), "B": ("L", 1350, 700, 1.6)}
    msgs = [m for _, m in catalogue_alarms(z, places)]
    assert "tête de A trop près du bord haut" in msgs
    msgs = [
        m for _, m in catalogue_alarms({"B": ("L", 1350, 700, 1.2)}, {"B": places["B"]})
    ]
    assert msgs[0] == "trop d'air au-dessus des têtes"


def test_alarm_log_and_draw(tmp_path, pictures):
    alarms = [
        Alarm(140.84, 145.0, "Fanfan", "tête", "tête de Fanfan trop près du bord haut")
    ]
    log = alarm_log(
        alarms, tmp_path / "alarmes.md", locate=lambda t: "mesure 61, couplet 2"
    )
    text = log.read_text(encoding="utf-8")
    assert (
        "| 2:20,8, mesure 61, couplet 2 | Fanfan | tête : tête de Fanfan trop près du bord haut |  |  |"
        in text
    )
    with pytest.raises(FileExistsError):
        alarm_log(alarms, log)
    alarm_log([], log, overwrite=True)
    assert clock(59.96) == "1:00,0"
    frame = {"people": [{**body(500), "name": "T"}, body(1500)]}
    dst = draw_frame(pictures[0], ("L", 960, 540, 2.0), tmp_path / "cadre.jpg", frame)
    assert read_image(dst).shape == (1080, 1920, 3)
