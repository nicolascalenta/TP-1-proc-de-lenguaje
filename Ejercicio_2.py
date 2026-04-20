import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

ANSWER_KEY = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']


def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    return img


def binarize(gray: np.ndarray):
    """Binarización global con Otsu."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def normalize_char(char_bw: np.ndarray, size: Tuple[int, int] = (32, 32)):
    ys, xs = np.where(char_bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(size, dtype=np.uint8)

    char_bw = char_bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = char_bw.shape
    side = max(h, w) + 6
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = char_bw
    return cv2.resize(canvas, size, interpolation=cv2.INTER_NEAREST)


def pixel_diff(a: np.ndarray, b: np.ndarray):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def count_holes(char_bw: np.ndarray):
    contornos, hierarchy = cv2.findContours(char_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    for h in hierarchy[0]:
        if h[3] != -1:
            holes += 1
    return holes

def order_question_boxes(boxes: List[Tuple[int, int, int, int]]):
    """
    Ordena las cajas.
    """
    if len(boxes) != 10:
        return sorted(boxes, key=lambda box: (box[0], box[1]))

    x_centers = np.array([x + w / 2 for x, y, w, h in boxes], dtype=np.float32)
    x_split = float(np.median(x_centers))

    left_col = [b for b in boxes if (b[0] + b[2] / 2) < x_split]
    right_col = [b for b in boxes if (b[0] + b[2] / 2) >= x_split]

    left_col.sort(key=lambda box: box[1])
    right_col.sort(key=lambda box: box[1])

    return left_col + right_col

def find_question_boxes(bw: np.ndarray):
    """
    Busca los 10 rectángulos grandes de las preguntas.
    """
    contornos, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 220 < w < 260 and 110 < h < 140 and area > 25000:
            boxes.append((x, y, w, h))

    unique_boxes = []
    for b in boxes:
        is_duplicate = any(
            abs(b[0] - u[0]) < 3 and
            abs(b[1] - u[1]) < 3 and
            abs(b[2] - u[2]) < 3 and
            abs(b[3] - u[3]) < 3
            for u in unique_boxes
        )
        if not is_duplicate:
            unique_boxes.append(b)

    return order_question_boxes(unique_boxes)

def crop_questions(gray: np.ndarray, boxes: List[Tuple[int, int, int, int]], margin: int = 2):
    questions = []
    for x, y, w, h in boxes:
        questions.append(gray[y + margin:y + h - margin, x + margin:x + w - margin])
    return questions

def group_option_lines(question_bw: np.ndarray):
    """
    Dentro de la mitad inferior izquierda de una pregunta, agrupa componentes por línea
    y toma la componente más a la izquierda, que corresponde a la letra A/B/C/D.
    """
    region = question_bw[int(question_bw.shape[0] * 0.45):question_bw.shape[0] - 3, 3:int(question_bw.shape[1] * 0.50)]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(region, 8, cv2.CV_32S)

    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 10 and h >= 6 and x > 2:
            components.append((x, y, w, h, area))

    line_groups = []
    centers = []
    for comp in sorted(components, key=lambda item: item[1]):
        cy = comp[1] + comp[3] / 2
        assigned = False
        for k, line_y in enumerate(centers):
            if abs(cy - line_y) < 8:
                line_groups[k].append(comp)
                centers[k] = float(np.mean([c[1] + c[3] / 2 for c in line_groups[k]]))
                assigned = True
                break
        if not assigned:
            line_groups.append([comp])
            centers.append(cy)

    grouped = sorted(zip(centers, line_groups), key=lambda item: item[0])

    leftmost_components = []
    for _, line in grouped[:4]:
        leftmost_components.append(sorted(line, key=lambda item: item[0])[0])

    if len(leftmost_components) != 4:
        raise RuntimeError("No se pudieron extraer las 4 plantillas A/B/C/D del examen.")

    return region, leftmost_components


def respuestas_tmpl(question_gray: np.ndarray):
    question_bw = binarize(question_gray)
    region, letter_boxes = group_option_lines(question_bw)
    labels = ['A', 'B', 'C', 'D']
    tmpl = {}

    for label, (x, y, w, h, _) in zip(labels, letter_boxes):
        char_bw = region[y:y + h, x:x + w]
        tmpl[label] = normalize_char(char_bw)

    return tmpl


def detect_underline(question_bw: np.ndarray):
    """Busca la línea horizontal larga donde se escribe la respuesta."""
    inner = question_bw[3:-3, 3:-3]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inner, 8, cv2.CV_32S)

    best = None
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        is_good_line = (
            y < inner.shape[0] * 0.45 and
            w > 40 and
            h <= 3 and
            area >= 40
        )
        if is_good_line and (best is None or w > best[2]):
            best = (x, y, w, h, area)

    return best


def extract_answer(question_gray: np.ndarray):
    question_bw = binarize(question_gray)
    underline = detect_underline(question_bw)
    if underline is None:
        raise RuntimeError("No se pudo detectar la respuesta")

    x, y, w, h, _ = underline
    x += 3
    y += 3

    cx = x + w // 2
    x1 = max(0, cx - 28)
    x2 = min(question_gray.shape[1], cx + 28)
    y1 = max(0, y - 20)
    y2 = min(question_gray.shape[0], y + 4)

    return question_gray[y1:y2, x1:x2]


def segment_answer_chars(answer_gray: np.ndarray):
    """
    Dentro de la roi de respuesta elimina la línea y conserva sólo caracteres candidatos.
    """
    bw = binarize(answer_gray)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)

    chars = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        is_line = (w > bw.shape[1] * 0.5 and h <= 3)
        if is_line:
            continue

        if area < 12 or h < 6 or w < 3:
            continue

        cy = y + h / 2
        if cy < bw.shape[0] * 0.35:
            continue

        chars.append((x, y, w, h, area))

    chars.sort(key=lambda item: item[0])
    return bw, chars


def clasificar_answer(char_bw: np.ndarray, tmpl: Dict[str, np.ndarray]):
    "definimos en base a los agujeros en los contornos si la respuesta es A, B, C o D"
    char_norm = normalize_char(char_bw)
    holes = count_holes(char_norm)

    if holes == 0:
        return 'C'
    if holes >= 2:
        return 'B'

    candidate_labels = ['A', 'D']
    scores = {label: pixel_diff(char_norm, tmpl[label]) for label in candidate_labels}
    return min(scores, key=scores.get)


def read_answer(question_gray: np.ndarray, tmpl: Dict[str, np.ndarray]):
    answer_roi = extract_answer(question_gray)
    answer_bw, chars = segment_answer_chars(answer_roi)

    if len(chars) != 1:
        return None

    x, y, w, h, _ = chars[0]
    char_bw = answer_bw[y:y + h, x:x + w]
    return clasificar_answer(char_bw, tmpl)


def find_campos_encabezado(gray: np.ndarray):
    """
    Detecta las tres líneas del encabezado y genera recortes de los campos.
    """
    top_h = int(gray.shape[0] * 0.16)
    header_gray = gray[:top_h, :]
    header_bw = binarize(header_gray)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(header_bw, 8, cv2.CV_32S)
    lines = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w > 35 and h <= 3 and y < top_h * 0.75:
            lines.append((x, y, w, h, area))

    lines.sort(key=lambda item: item[0])

    if len(lines) < 3:
        raise RuntimeError("No se pudieron detectar las tres líneas del encabezado.")

    lines = sorted(lines, key=lambda item: item[2], reverse=True)[:3]
    lines = sorted(lines, key=lambda item: item[0])

    campos = ['nombre', 'fecha', 'clase']
    fields = {}
    for campo, (x, y, w, h, _) in zip(campos, lines):
        x1 = max(0, x - 2)
        x2 = min(header_gray.shape[1], x + w + 2)
        y1 = max(0, y - 22)
        y2 = min(header_gray.shape[0], y + 4)
        fields[campo] = header_gray[y1:y2, x1:x2]

    return fields


def segment_char(field_gray: np.ndarray):
    "Se segmentan los caracteres de un elemento"
    bw = binarize(field_gray)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    chars = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if w > bw.shape[1] * 0.40 and h <= 3:
            continue

        if area < 4 or h < 6 or w < 1:
            continue

        chars.append((x, y, w, h, area))

    chars.sort(key=lambda item: item[0])
    return chars


def cuenta_palabras(char_boxes: List[Tuple[int, int, int, int, int]]):
    if not char_boxes:
        return 0

    widths = [box[2] for box in char_boxes]
    width_media = float(np.median(widths)) if widths else 0.0
    gap_threshold = max(3.0, 0.5 * width_media)

    palabras = 1
    for box_anterior, box_actual in zip(char_boxes[:-1], char_boxes[1:]):
        prev_end = box_anterior[0] + box_anterior[2]
        gap = box_actual[0] - prev_end
        if gap > gap_threshold:
            palabras += 1

    return palabras


def validar_encabezado(campo: str, field_gray: np.ndarray):
    chars = segment_char(field_gray)
    char_count = len(chars)
    word_count = cuenta_palabras(chars)

    if campo == 'nombre':
        ok = (word_count >= 2) and (char_count <= 25)
    elif campo == 'fecha':
        ok = (char_count == 8)
    elif campo == 'clase':
        ok = (char_count == 1)
    else:
        raise ValueError(f"Campo no soportado: {campo}")

    return ok, {
        'char_count': char_count,
        'word_count': word_count,
    }

def evaluar_encabezado(gray: np.ndarray):
    campos_encabezado = find_campos_encabezado(gray)
    encabezado = {}

    for campo, field_gray in campos_encabezado.items():
        ok, info = validar_encabezado(campo, field_gray)
        encabezado[campo] = {
            'ok': ok,
            **info,
            'recorte': field_gray,
        }

    return encabezado


def evaluar_answers(gray: np.ndarray, bw: np.ndarray):
    question_boxes = find_question_boxes(bw)
    questions = crop_questions(gray, question_boxes)
    tmpl = respuestas_tmpl(questions[0])

    preguntas = []
    correctas = 0

    for i, question_gray in enumerate(questions, start=1):
        detected = read_answer(question_gray, tmpl)
        expected = ANSWER_KEY[i - 1]
        is_ok = (detected == expected)
        correctas += int(is_ok)

        preguntas.append({
            'question': i,
            'detected': detected,
            'expected': expected,
            'ok': is_ok,
        })

    return preguntas, correctas

def corregir_examen(image_path: str):
    gray = load_gray(image_path)
    bw = binarize(gray)
    encabezado = evaluar_encabezado(gray)
    preguntas, correctas = evaluar_answers(gray, bw)
    aprobo = correctas >= 6

    return {
        'image_path': image_path,
        'encabezado': encabezado,
        'preguntas': preguntas,
        'correctas': correctas,
        'aprobo': aprobo,
        'recorte_nombre': encabezado['nombre']['recorte'],
    }


def imagen_resultados(resultados: List[Dict], output_path: str = 'resumen_examenes.png'):
    recorte_nombres = [res['recorte_nombre'] for res in resultados]

    target_h = max(recorte.shape[0] for recorte in recorte_nombres) + 12
    target_w = max(recorte.shape[1] for recorte in recorte_nombres) + 180

    row_images = []
    for res, recorte in zip(resultados, recorte_nombres):
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

        y0 = (target_h - recorte.shape[0]) // 2
        x0 = 8
        canvas[y0:y0 + recorte.shape[0], x0:x0 + recorte.shape[1]] = recorte_rgb

        texto = 'APROBADO' if res['aprobo'] else 'DESAPROBADO'

        cv2.rectangle(
            canvas,
            (x0 - 4, y0 - 4),
            (x0 + recorte.shape[1] + 4, y0 + recorte.shape[0] + 4),
            (0, 0, 0),
            2,
        )

        cv2.putText(
            canvas,
            texto,
            (recorte.shape[1] + 24, target_h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        row_images.append(canvas)

    resumen = np.vstack(row_images)
    cv2.imwrite(output_path, resumen)
    return output_path

def print_resultados(result: Dict):
    archivo = os.path.basename(result['image_path'])
    print(f"\n{'=' * 60}")
    print(f"Examen: {archivo}")
    print('-' * 60)

    print("ENCABEZADO")
    for campo in ['nombre', 'fecha', 'clase']:
        info = result['encabezado'][campo]
        status = 'OK' if info['ok'] else 'MAL'
        print(f"{campo}: {status} (chars={info['char_count']}, palabras={info['word_count']})")

    print('-' * 60)
    print("RESPUESTAS")
    for item in result['preguntas']:
        status = 'OK' if item['ok'] else 'MAL'
        detected = item['detected'] if item['detected'] is not None else 'SIN/VARIAS'
        print(f"Pregunta {item['question']:02d}: {status} | detectada={detected} | correcta={item['expected']}")

    print('-' * 60)
    print(f"Total correctas: {result['correctas']}/10")
    print(f"Estado final: {'APROBADO' if result['aprobo'] else 'DESAPROBADO'}")

def run_batch(image_paths: List[str], resumen_output: str = 'resumen_examenes.png'):
    resultados = []
    for image_path in image_paths:
        result = corregir_examen(image_path)
        print_resultados(result)
        resultados.append(result)

    resumen_path = imagen_resultados(resultados, resumen_output)
    print(f"\nImagen resumen generada en: {resumen_path}")
    return resultados

def seleccionar_examenes():
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_paths = filedialog.askopenfilenames(
        title='Seleccionar exámenes a corregir',
        filetypes=[
            ('Imágenes', '*.png *.jpg *.jpeg *.tif *.tiff *.bmp'),
            ('PNG', '*.png'),
            ('Todos los archivos', '*.*'),
        ]
    )

    root.destroy()

    exam_files = [str(Path(p)) for p in file_paths]

    if not exam_files:
        raise RuntimeError('No se seleccionaron archivos.')

    return exam_files

if __name__ == '__main__':
    exam_files = seleccionar_examenes()

    output_dir = Path(exam_files[0]).resolve().parent
    resumen_output = output_dir / 'resumen.png'

    run_batch(
        exam_files,
        resumen_output=str(resumen_output)
    )
