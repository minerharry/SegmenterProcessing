# NOTE: This file was automatically generated from:
# C:\Users\miner\Documents\GitHub\segmenterProcessing\image_annotations.py
# DO NOT CHANGE DIRECTLY! 1743365985.3276463
try:
    timedelta, = ultraimport('__dir__/datetime/__init__.py', objects_to_import=('timedelta',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        timedelta, = ultraimport('__dir__/datetime.py', objects_to_import=('timedelta',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from datetime import timedelta', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 1, 0), object_to_import='timedelta', combine=[e, e2]) from None
import functools
import itertools
try:
    Callable, = ultraimport('__dir__/typing/__init__.py', objects_to_import=('Callable',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        Callable, = ultraimport('__dir__/typing.py', objects_to_import=('Callable',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from typing import Callable, Concatenate, Iterable, overload', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 4, 0), object_to_import='Callable', combine=[e, e2]) from None
try:
    Concatenate, = ultraimport('__dir__/typing/__init__.py', objects_to_import=('Concatenate',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        Concatenate, = ultraimport('__dir__/typing.py', objects_to_import=('Concatenate',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from typing import Callable, Concatenate, Iterable, overload', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 4, 0), object_to_import='Concatenate', combine=[e, e2]) from None
try:
    Iterable, = ultraimport('__dir__/typing/__init__.py', objects_to_import=('Iterable',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        Iterable, = ultraimport('__dir__/typing.py', objects_to_import=('Iterable',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from typing import Callable, Concatenate, Iterable, overload', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 4, 0), object_to_import='Iterable', combine=[e, e2]) from None
try:
    overload, = ultraimport('__dir__/typing/__init__.py', objects_to_import=('overload',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        overload, = ultraimport('__dir__/typing.py', objects_to_import=('overload',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from typing import Callable, Concatenate, Iterable, overload', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 4, 0), object_to_import='overload', combine=[e, e2]) from None
import cv2
import numpy as np
try:
    Metadata, = ultraimport('__dir__/utils.metadata/__init__.py', objects_to_import=('Metadata',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        Metadata, = ultraimport('__dir__/utils.metadata.py', objects_to_import=('Metadata',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from utils.metadata import Metadata', 'C:\\Users\\miner\\Documents\\GitHub\\segmenterProcessing\\image_annotations.py', 8, 0), object_to_import='Metadata', combine=[e, e2]) from None
textfont = cv2.FONT_HERSHEY_SIMPLEX

def applicable[**P, **P2, R: Callable[[np.ndarray, int], np.ndarray]](f: Callable[Concatenate[tuple[int, ...], P], R]):

    @overload
    def apply(image: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        ...

    @overload
    def apply(image: Iterable[np.ndarray], *args: P.args, **kwargs: P.kwargs) -> 'map[np.ndarray]':
        ...

    def apply(image: np.ndarray | Iterable[np.ndarray], *args: P.args, **kwargs: P.kwargs):
        single = False
        if isinstance(image, np.ndarray):
            single = True
            image = [image]
        it = iter(image)
        im = next(it)
        it = itertools.chain([im], it)
        func = f(im.shape, *args, **kwargs)
        if single:
            return func(next(it), 0)
        else:
            return map(func, it, itertools.count())
    return apply

@applicable
def draw_scalebar(shape: tuple[int, ...], pixelscale: float | Metadata, scaleLength: float=250, scaleUnit: str | None='um', textsize: float=1, scaleWidth: int=5, color: tuple[int, int, int]=(255, 255, 255), thickness: int=2):
    length = scaleLength / pixelscale
    text = f'{scaleLength} {scaleUnit}'
    (label_width, label_height), baseline = cv2.getTextSize(text, textfont, textsize, 1)
    text_height = label_height + baseline
    scale_top = 5 + text_height + 3
    rect = [(shape[1] - 5, scale_top), (int(shape[1] - 5 - length), scale_top + scaleWidth)]
    textcorner = (int(shape[1] - 5 - length / 2 - label_width / 2), 5 + label_height)

    def _apply(im, frame: int):
        im = cv2.putText(im.copy(), text, textcorner, textfont, textsize, color, thickness)
        im = cv2.rectangle(im, *rect, color, -1)
        return im
    return _apply

@applicable
def draw_timestamp(shape: tuple[int, ...], delta: timedelta | None=None, textsize: float=1, bigformat: bool=False, color: tuple[int, int, int]=(255, 255, 255), thickness: int=2):

    def _apply(im, frame: int):
        if delta:
            real = (delta * frame).total_seconds()
            if bigformat:
                hour, minute, second = (int(real // 3600), int(real % 3600 // 60), real % 60)
                time = f'{hour}h {minute}m'
            else:
                minute, second = (int(real // 60), real % 60)
                time = f'{minute}:{second:02.2f}'
        else:
            time = str(frame)
        return cv2.putText(im.copy(), time, (0, im.shape[0]), textfont, textsize, color, thickness)
    return _apply