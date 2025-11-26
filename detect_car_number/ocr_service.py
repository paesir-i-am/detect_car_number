from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytesseract


MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3


def _find_chars(contour_list: List[dict], possible_contours: List[dict]) -> List[List[int]]:
    """기하 조건을 만족하는 문자 후보들을 그룹핑."""
    matched_result_idx: List[List[int]] = []

    for d1 in contour_list:
        matched_contours_idx: List[int] = []
        for d2 in contour_list:
            if d1["idx"] == d2["idx"]:
                continue

            dx = abs(d1["cx"] - d2["cx"])
            dy = abs(d1["cy"] - d2["cy"])

            diagonal_length1 = np.sqrt(d1["w"] ** 2 + d1["h"] ** 2)
            distance = np.linalg.norm(np.array([d1["cx"], d1["cy"]]) - np.array([d2["cx"], d2["cy"]]))
            angle_diff = 90 if dx == 0 else np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1["w"] * d1["h"] - d2["w"] * d2["h"]) / (d1["w"] * d1["h"])
            width_diff = abs(d1["w"] - d2["w"]) / d1["w"]
            height_diff = abs(d1["h"] - d2["h"]) / d1["h"]

            if (
                distance < diagonal_length1 * MAX_DIAG_MULTIPLYER
                and angle_diff < MAX_ANGLE_DIFF
                and area_diff < MAX_AREA_DIFF
                and width_diff < MAX_WIDTH_DIFF
                and height_diff < MAX_HEIGHT_DIFF
            ):
                matched_contours_idx.append(d2["idx"])

        matched_contours_idx.append(d1["idx"])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = [d4["idx"] for d4 in contour_list if d4["idx"] not in matched_contours_idx]
        if unmatched_contour_idx:
            unmatched_contour = list(np.take(possible_contours, unmatched_contour_idx))
            recursive_contour_list = _find_chars(unmatched_contour, possible_contours)
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
        break

    return matched_result_idx


def tesseract_ocr(img_path: Path, tesseract_cmd: Path, tessdata_prefix: Path) -> str:
    """크롭된 번호판 이미지에서 한국어/숫자 문자열을 추출."""
    possible_contours: List[dict] = []

    img_array = np.fromfile(str(img_path), np.uint8)
    img_ori = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_ori is None:
        return ""

    height, width, channel = img_ori.shape
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9,
    )

    contours, _ = cv2.findContours(img_blur_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append(
            {
                "contour": contour,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + (w / 2),
                "cy": y + (h / 2),
            }
        )

    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    cnt = 0
    for d in contours_dict:
        area = d["w"] * d["h"]
        ratio = d["w"] / d["h"]

        if area > MIN_AREA and d["w"] > MIN_WIDTH and d["h"] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d["idx"] = cnt
            cnt += 1
            possible_contours.append(d)

    result_idx = _find_chars(possible_contours, possible_contours)

    matched_result = [list(np.take(possible_contours, idx_list)) for idx_list in result_idx]

    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for matched_chars in matched_result:
        sorted_chars = sorted(matched_chars, key=lambda x: x["cx"])

        plate_cx = (sorted_chars[0]["cx"] + sorted_chars[-1]["cx"]) / 2
        plate_cy = (sorted_chars[0]["cy"] + sorted_chars[-1]["cy"]) / 2

        plate_width = (sorted_chars[-1]["x"] + sorted_chars[-1]["w"] - sorted_chars[0]["x"]) * PLATE_WIDTH_PADDING
        sum_height = sum(d["h"] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]["cy"] - sorted_chars[0]["cy"]
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]["cx"], sorted_chars[0]["cy"]])
            - np.array([sorted_chars[-1]["cx"], sorted_chars[-1]["cy"]])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated, patchSize=(int(plate_width), int(plate_height)), center=(int(plate_cx), int(plate_cy))
        )

        ratio = img_cropped.shape[1] / img_cropped.shape[0] if img_cropped.shape[0] > 0 else 0
        if ratio < MIN_PLATE_RATIO or ratio > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append(
            {
                "x": int(plate_cx - plate_width / 2),
                "y": int(plate_cy - plate_height / 2),
                "w": int(plate_width),
                "h": int(plate_height),
            }
        )

    result_chars = ""
    for plate_img in plate_imgs:
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h
            if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
                plate_min_x = min(plate_min_x, x)
                plate_min_y = min(plate_min_y, y)
                plate_max_x = max(plate_max_x, x + w)
                plate_max_y = max(plate_max_y, y + h)

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(
            img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        os.environ["TESSDATA_PREFIX"] = str(tessdata_prefix)
        chars = pytesseract.image_to_string(img_result, lang="kor", config="--psm 7 --oem 0")

        filtered = ""
        for c in chars:
            if ord("가") <= ord(c) <= ord("힣") or c.isdigit():
                filtered += c

        if len(filtered) > len(result_chars):
            result_chars = filtered

    return result_chars
