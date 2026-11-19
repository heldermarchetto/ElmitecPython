# -*- coding: utf-8 -*-
"""
elmitecAnalysis - browse and analyse stacks of Elmitec uView images (.dat).

MIT License

Copyright (c) 2019 Dr. Helder Marchetto (modernised 2026)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Usage
-----
From the command line (opens a file dialog if no files are given)::

    python elmitecAnalysis.py K:\\Data\\TurningPointResolution
    python elmitecAnalysis.py img001.dat img002.dat img003.dat

From Python / Spyder (in Spyder first run ``%matplotlib qt`` so the window is
interactive; in Jupyter use ``%matplotlib widget``)::

    from elmitecAnalysis import imgStackClass, elmitecImageViewer, getDatFilesInDir

    stack = imgStackClass(r"K:\\Data\\TurningPointResolution")   # folder or list of files
    viewer = elmitecImageViewer(stack)

    img  = stack.getImage(5)          # numpy array of image 5
    data = stack.asArray()            # whole stack as (n, height, width) array
    stack.getParameter(5, "Start Voltage")

Viewer controls
---------------
* Image slider, or the ◀ / ▶ buttons
* Keyboard: ← → (or ↑ ↓) one image, PageUp / PageDown 10 images, Home / End,
  space = play / pause, A = toggle auto contrast
* Mouse wheel over the window: previous / next image
* Contrast slider: drag both ends; "Auto contrast" clips 2 % / 98 % per image
* The status bar shows the pixel position and intensity under the mouse
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider

import readUview as ru

__all__ = ["imgStackClass", "elmitecImageViewer", "getDatFilesInDir"]
__version__ = "2.0.0"

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _natural_key(path: str) -> list:
    """Sort key so that 'img2.dat' comes before 'img10.dat'."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def getDatFilesInDir(mypath: PathLike) -> List[str]:
    """Return all .dat files in *mypath*, sorted naturally."""
    mypath = os.fspath(mypath)
    if not os.path.isdir(mypath):
        raise FileNotFoundError(f"Folder not found: {mypath}")
    files = [os.path.join(mypath, f) for f in os.listdir(mypath)
             if f.lower().endswith(".dat")]
    return sorted(files, key=_natural_key)


# --------------------------------------------------------------------------- #
# Image stack
# --------------------------------------------------------------------------- #
class imgStackClass:
    """A stack of uView images, loaded on demand.

    Parameters
    ----------
    fn : folder name, or list of file names
    preload : bool
        Read all images immediately (otherwise they are read when needed).
    cacheSize : int
        Number of images kept in memory when not preloading.
    """

    def __init__(self, fn: Union[PathLike, Sequence[PathLike]], preload: bool = False,
                 cacheSize: int = 64) -> None:
        if isinstance(fn, (str, os.PathLike)):
            fn = getDatFilesInDir(fn) if os.path.isdir(fn) else [fn]
        self.fn: List[str] = [os.fspath(f) for f in fn]
        if not self.fn:
            raise ValueError("No image files given")

        self.nImages = len(self.fn)
        self.rawfn = [os.path.basename(f) for f in self.fn]
        self.dir = os.path.dirname(os.path.abspath(self.fn[0]))
        self.current = 0
        self.cacheSize = self.nImages if preload else max(1, cacheSize)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, List[dict]]]" = OrderedDict()

        first = self.getImage(0)
        self.imageHeight, self.imageWidth = first.shape
        self.imgType = first.dtype.name
        self.limits = self.getLimits(0)

        if preload:
            for i in range(self.nImages):
                self._load(i)
                print(f"\rLoading image {i + 1:4d}/{self.nImages}", end="", flush=True)
            print()

    # -- container behaviour ------------------------------------------------ #
    def __len__(self) -> int:
        return self.nImages

    def __getitem__(self, pos: int) -> np.ndarray:
        return self.getImage(pos)

    def __iter__(self):
        return (self.getImage(i) for i in range(self.nImages))

    def __repr__(self) -> str:
        return "\n".join([
            f"nImages = {self.nImages:04d}",
            f"First image = {self.rawfn[0]}",
            f"Last image = {self.rawfn[-1]}",
            f"Directory = {self.dir}",
            f"Image size = ({self.imageWidth},{self.imageHeight})",
        ])

    # -- loading ------------------------------------------------------------- #
    def _index(self, pos: int) -> int:
        if pos is None or pos < 0:
            pos = self.current
        pos = int(pos)
        if not 0 <= pos < self.nImages:
            raise IndexError(f"Image index {pos} out of range 0..{self.nImages - 1}")
        return pos

    def _load(self, pos: int) -> Tuple[np.ndarray, List[dict]]:
        if pos in self._cache:
            self._cache.move_to_end(pos)
            return self._cache[pos]
        reader = ru.readUviewClass()
        try:
            img = reader.getImage(self.fn[pos])
        except ru.UviewFileError as exc:
            raise ru.UviewFileError(f"Image {pos} ({self.rawfn[pos]}): {exc}") from exc
        entry = (img, list(reader.paramList))
        self._cache[pos] = entry
        while len(self._cache) > self.cacheSize:
            self._cache.popitem(last=False)
        return entry

    # -- public API ------------------------------------------------------------ #
    def getImage(self, pos: int = -1) -> np.ndarray:
        """Return image *pos* (default: the current one) as numpy array."""
        return self._load(self._index(pos))[0]

    def getParamList(self, pos: int = -1) -> List[dict]:
        """Return the LEEM parameters of image *pos* as a list of dicts."""
        return self._load(self._index(pos))[1]

    def getParameter(self, pos: int, name: str, default: Any = None) -> Any:
        """Return the value of LEEM parameter *name* (case-insensitive) for image *pos*."""
        key = name.strip().lower()
        for p in self.getParamList(pos):
            if p["name"].strip().lower() == key:
                return p["value"]
        return default

    def getLimits(self, pos: int = -1, clip: float = 2) -> np.ndarray:
        """Intensity limits that clip *clip* % at both ends of the histogram."""
        self.limits = np.percentile(self.getImage(pos), (clip, 100 - clip))
        return self.limits

    def getDrawImage(self, pos: int = -1, clip: float = 2) -> np.ndarray:
        """Return image *pos* scaled to 0..255 (uint8) with *clip* % contrast clipping."""
        img = self.getImage(pos).astype(np.float32)
        lo, hi = np.percentile(img, (clip, 100 - clip))
        scaled = (img - lo) / max(hi - lo, 1e-12)
        return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

    def asArray(self) -> np.ndarray:
        """Return all images as one array of shape (nImages, height, width)."""
        return np.stack([self.getImage(i) for i in range(self.nImages)])


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #
# Keys used by the viewer that matplotlib would otherwise use for its toolbar
_VIEWER_KEYS = {"left", "right", "up", "down", "home", "end", "pageup",
                "pagedown", " ", "a", "backspace"}


def _free_keys() -> None:
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in _VIEWER_KEYS]


class elmitecImageViewer:
    """Interactive viewer for an imgStackClass.

    Parameters
    ----------
    stack : imgStackClass, folder name, list of files, or None (opens a file dialog)
    clip : float
        Percent clipped at both ends for automatic contrast.
    show : bool
        Show the window immediately (set False when embedding or testing).
    """

    PLAY_INTERVAL_MS = 200
    MAX_PARAMS_SHOWN = 30

    def __init__(self, stack: Union[imgStackClass, PathLike, Sequence[PathLike], None] = None,
                 clip: float = 2.0, show: bool = True) -> None:
        if stack is None:
            files = self._ask_files()
            if not files:
                raise ValueError("No files selected")
            stack = files
        if not isinstance(stack, imgStackClass):
            stack = imgStackClass(stack)
        self.stack = stack
        self.clip = clip
        self.autoContrast = True
        self._updating = False          # guards against widget callback loops
        self._playing = False

        _free_keys()
        self._build_figure()
        self._configure_for_stack()
        self.showImage(self.stack.current)
        if show:
            plt.show()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(11, 8))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Elmitec image stack viewer")

        self.ax = self.fig.add_axes((0.04, 0.20, 0.62, 0.74))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.im = self.ax.imshow(np.zeros((2, 2)), cmap="gray", interpolation="nearest")
        self.ax.format_coord = self._format_coord

        self.infoAx = self.fig.add_axes((0.69, 0.20, 0.30, 0.74))
        self.infoAx.axis("off")
        self.infoText = self.infoAx.text(0, 1, "", va="top", ha="left",
                                         family="monospace", fontsize=8,
                                         transform=self.infoAx.transAxes)

        self.slider = Slider(self.fig.add_axes((0.12, 0.11, 0.44, 0.035)), "Image",
                             0, 1, valinit=0, valstep=1, valfmt="%d")
        self.slider.on_changed(self._on_slider)

        self.contrast = RangeSlider(self.fig.add_axes((0.12, 0.05, 0.44, 0.035)),
                                    "Contrast", 0, 1, valinit=(0, 1), valfmt="%.0f")
        self.contrast.on_changed(self._on_contrast)

        self.btnPrev = Button(self.fig.add_axes((0.69, 0.10, 0.06, 0.05)), "◀")
        self.btnNext = Button(self.fig.add_axes((0.76, 0.10, 0.06, 0.05)), "▶")
        self.btnPlay = Button(self.fig.add_axes((0.83, 0.10, 0.07, 0.05)), "Play")
        self.btnOpen = Button(self.fig.add_axes((0.91, 0.10, 0.07, 0.05)), "Open…")
        self.btnPrev.on_clicked(lambda _e: self.step(-1))
        self.btnNext.on_clicked(lambda _e: self.step(+1))
        self.btnPlay.on_clicked(lambda _e: self.togglePlay())
        self.btnOpen.on_clicked(lambda _e: self.openImageList())

        self.chkAuto = CheckButtons(self.fig.add_axes((0.69, 0.03, 0.20, 0.06)),
                                    ["Auto contrast"], [self.autoContrast])
        self.chkAuto.on_clicked(self._on_check)

        self.timer = self.fig.canvas.new_timer(interval=self.PLAY_INTERVAL_MS)
        self.timer.add_callback(self._on_timer)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _configure_for_stack(self) -> None:
        """Adapt slider ranges to the current stack."""
        n = self.stack.nImages
        self._updating = True
        try:
            # Slider needs valmax > valmin, even for a single image
            self.slider.valmax = max(n - 1, 1)
            self.slider.ax.set_xlim(0, self.slider.valmax)
            self.slider.set_val(min(self.stack.current, n - 1))
            self.slider.set_active(n > 1)

            img = self.stack.getImage(0)
            lo, hi = float(img.min()), float(img.max())
            self._set_contrast_range(lo, hi + max(0.2 * (hi - lo), 1.0))
        finally:
            self._updating = False

    def _set_contrast_range(self, vmin: float, vmax: float) -> None:
        self.contrast.valmin, self.contrast.valmax = vmin, vmax
        self.contrast.ax.set_xlim(vmin, vmax)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def showImage(self, imgNr: int = -1) -> None:
        """Display image *imgNr* (default: the current one) and update all widgets."""
        n = self.stack.nImages
        idx = self.stack.current if imgNr is None or imgNr < 0 else int(imgNr)
        idx = min(max(idx, 0), n - 1)
        self.stack.current = idx

        try:
            img = self.stack.getImage(idx)
        except (ru.UviewFileError, OSError) as exc:
            log.error("%s", exc)
            self.ax.set_title(f"{idx:04d}: cannot read {self.stack.rawfn[idx]}", color="red")
            self.fig.canvas.draw_idle()
            return

        h, w = img.shape
        self.im.set_data(img)
        self.im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)

        self._updating = True
        try:
            if self.autoContrast:
                lo, hi = (float(v) for v in np.percentile(img, (self.clip, 100 - self.clip)))
                if hi <= lo:
                    hi = lo + 1
                if lo < self.contrast.valmin or hi > self.contrast.valmax:
                    self._set_contrast_range(min(lo, self.contrast.valmin),
                                             max(hi, self.contrast.valmax))
                self.contrast.set_val((lo, hi))
            else:
                lo, hi = self.contrast.val
            self.im.set_clim(lo, hi)
            if int(self.slider.val) != idx:
                self.slider.set_val(idx)
        finally:
            self._updating = False

        self.ax.set_title(f"{idx:04d} / {n - 1:04d}   {self.stack.rawfn[idx]}",
                          fontsize=10, color="black")
        self.infoText.set_text(self._info_text(idx, img))
        self.fig.canvas.draw_idle()

    def _info_text(self, idx: int, img: np.ndarray) -> str:
        lines = [
            f"File : {self.stack.rawfn[idx]}",
            f"Size : {img.shape[1]} x {img.shape[0]}",
            f"Min / max : {img.min()} / {img.max()}",
            f"Mean : {img.mean():.1f}",
            "",
            "LEEM parameters",
            "---------------",
        ]
        try:
            params = self.stack.getParamList(idx)
        except (ru.UviewFileError, OSError):
            params = []
        for p in params[: self.MAX_PARAMS_SHOWN]:
            value = p["value"]
            value = f"{value:.4g}" if isinstance(value, float) else str(value)
            units = "" if p["units"] in ("", "none") else p["units"]
            lines.append(f"{p['name'][:30]:<30} {value:>9} {units}")
        if len(params) > self.MAX_PARAMS_SHOWN:
            lines.append(f"... {len(params) - self.MAX_PARAMS_SHOWN} more")
        return "\n".join(lines)

    def _format_coord(self, x: float, y: float) -> str:
        col, row = int(round(x)), int(round(y))
        data = self.im.get_array()
        if data is not None and 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            return f"x={col}, y={row}, I={data[row, col]}"
        return f"x={col}, y={row}"

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    def step(self, delta: int) -> None:
        """Move *delta* images forward (or back if negative)."""
        self.showImage(min(max(self.stack.current + delta, 0), self.stack.nImages - 1))

    def togglePlay(self) -> None:
        """Start or stop playing the stack as a movie."""
        self._playing = not self._playing
        if self._playing:
            self.btnPlay.label.set_text("Pause")
            self.timer.start()
        else:
            self.btnPlay.label.set_text("Play")
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _on_timer(self) -> None:
        if self._playing:
            self.showImage((self.stack.current + 1) % self.stack.nImages)

    def _set_auto(self, state: bool, redraw: bool = True) -> None:
        """Switch automatic contrast on or off (keeps the checkbox in sync)."""
        self.autoContrast = state
        if self.chkAuto.get_status()[0] != state:
            self._updating = True
            try:
                self.chkAuto.set_active(0)      # fires _on_check, which is ignored
            finally:
                self._updating = False
        if redraw:
            self.showImage()

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    def _on_slider(self, val: float) -> None:
        if not self._updating:
            self.showImage(int(round(val)))

    def _on_check(self, _label: str) -> None:
        if not self._updating:
            self._set_auto(self.chkAuto.get_status()[0])

    def _on_contrast(self, val: Tuple[float, float]) -> None:
        if self._updating:
            return
        if self.autoContrast:              # manual change switches auto off
            self._set_auto(False, redraw=False)
        self.im.set_clim(*val)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        moves = {"right": 1, "down": 1, "left": -1, "up": -1,
                 "pagedown": 10, "pageup": -10}
        if event.key in moves:
            self.step(moves[event.key])
        elif event.key == "home":
            self.showImage(0)
        elif event.key == "end":
            self.showImage(self.stack.nImages - 1)
        elif event.key == " ":
            self.togglePlay()
        elif event.key == "a":
            self._set_auto(not self.autoContrast)

    def _on_scroll(self, event) -> None:
        self.step(-1 if event.button == "up" else 1)

    # ------------------------------------------------------------------ #
    # Opening files
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ask_files() -> List[str]:
        """Show a file dialog; return the chosen files (empty list if none)."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            print("No file dialog available (tkinter missing). "
                  "Use viewer.load([...file names...]) instead.")
            return []
        root = tk.Tk()
        root.withdraw()
        try:
            files = filedialog.askopenfilenames(
                title="Select uView images", filetypes=[("uView images", "*.dat"),
                                                        ("All files", "*.*")])
        finally:
            root.destroy()
        return sorted(files, key=_natural_key)

    def openImageList(self) -> None:
        """Choose new files with a dialog and show them."""
        if self._playing:
            self.togglePlay()
        files = self._ask_files()
        if files:
            self.load(files)

    def load(self, files: Union[imgStackClass, PathLike, Sequence[PathLike]]) -> None:
        """Replace the displayed stack (folder, list of files or imgStackClass)."""
        try:
            stack = files if isinstance(files, imgStackClass) else imgStackClass(files)
        except (ValueError, OSError, ru.UviewFileError) as exc:
            log.error("Could not open the images: %s", exc)
            return
        self.stack = stack
        self.stack.current = 0
        self._configure_for_stack()
        self.showImage(0)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = sys.argv[1:]
    if len(args) == 1 and os.path.isdir(args[0]):
        source: Any = args[0]
    elif args:
        source = args
    else:
        source = None                      # file dialog
    elmitecImageViewer(source)
