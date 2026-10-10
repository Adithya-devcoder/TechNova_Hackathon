"""SignBridge app: live webcam or uploaded video -> continuous ASL sign recognition.

Run from the repo root:
    streamlit run app/main.py
"""
import html
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.live_log import LOG_PATH, append_result, load_results, summarize  # noqa: E402
from models import pretrained_isolated  # noqa: E402
from models import signbridge_model as sb  # noqa: E402
from preprocessing.landmarks import HolisticExtractor, draw_landmarks, video_to_landmarks  # noqa: E402
from preprocessing.segmenter import SentenceSegmenter, resample, split_signs  # noqa: E402

DEMO_SENTENCES = json.loads((ROOT / 'data' / 'demo_sentences.json').read_text(encoding='utf-8'))
# Dataset sign names -> readable English, for display only
READABLE = {'minemy': 'my', 'weus': 'we', 'hesheit': 'he/she', 'thankyou': 'thank you', 'icecream': 'ice cream'}

st.set_page_config(page_title='SignBridge', page_icon='🤟', layout='wide')
CONTINUOUS = 'Continuous sentence: Kaggle 1st-place'
SIGN_BY_SIGN = 'Sign by sign: Kaggle 1st-place (pause between signs)'
HYBRID = 'Hybrid: ours + Kaggle 1st-place (continuous)'
OURS = 'Our model (BiGRU-CTC)'


@st.cache_resource
def load_bundle():
    return sb.load_model()


@st.cache_resource
def load_pretrained():
    return pretrained_isolated.IsolatedSignModel()


def readable(text):
    return ' '.join(READABLE.get(w, w) for w in text.split())


def words_html(result):
    """Recognised words as chips: green when confident, amber with a '?' when uncertain."""
    if not result['words']:
        return '<span style="opacity:.6">(no sign recognised)</span>'
    chips = []
    for w in result['words']:
        alts = ', '.join(f'{a} {p:.0%}' for a, p in w['alternatives'])
        tip = html.escape(f"confidence {w['confidence']:.0%} | alternatives: {alts}")
        if w['uncertain']:
            style = 'background:rgba(245,158,11,.22);border:1px dashed #d97706'
            label = html.escape(w['word']) + ' ?'
        else:
            style = 'background:rgba(22,163,74,.18);border:1px solid #16a34a'
            label = html.escape(w['word'])
        chips.append(f'<span title="{tip}" style="{style};border-radius:6px;padding:2px 8px;'
                     f'margin:2px;display:inline-block;font-size:1.25rem">{label}</span>')
    return ' '.join(chips)


def result_html(result, frames, reference=None):
    meta = f"{result['latency_ms']:.0f} ms · {frames} frames"
    line = f"<div style='margin:10px 0'>{words_html(result)}"
    if result['text']:
        line += f"<div style='opacity:.8;margin-top:4px'>“{html.escape(readable(result['text']))}”</div>"
    if reference is not None:
        mark = '✅' if result['text'] == reference else '❌'
        line += f"<div style='opacity:.8'>{mark} reference: {html.escape(reference)}</div>"
    return line + f"<small style='opacity:.6'>{meta}</small></div>"


# ---------------------------------------------------------------- sidebar
bundle_base = load_bundle()
with st.sidebar:
    st.title('🤟 SignBridge')
    st.caption('Continuous ASL sign-to-text (PS-41)')
    mode = st.radio('Mode', ['Live webcam', 'Upload video', 'Live evaluation'])
    st.subheader('Settings')
    cam_index = int(st.number_input('Camera index', 0, 5, 0))
    threshold = st.slider('Flag words below this confidence', 0.0, 1.0, 0.6, 0.05)
    pause_s = st.slider('Hands-down pause that ends a sentence (s)', 0.3, 2.0, 0.7, 0.1)
    target_fps = int(st.number_input('Model frame rate (fps)', 10, 60, 30,
                                     help='Sentences are resampled to this rate before recognition.'))
    show_overlay = st.checkbox('Draw landmarks on video', True)
    save_debug = st.checkbox('Save landmarks of each sentence (debugging)', False,
                             help='Saves only landmark coordinates (no video) to the local debug/ folder.')
    hybrid_ready = pretrained_isolated.available()
    recogniser = st.radio('Recogniser', [CONTINUOUS, SIGN_BY_SIGN, HYBRID, OURS], index=0 if hybrid_ready else 3,
                          disabled=not hybrid_ready,
                          help='Continuous: sign the whole sentence without pausing; the pretrained model reads it '
                               'with a sliding window. Sign by sign: lower your hands briefly between signs. '
                               'Kaggle 1st-place model. Hybrid: continuous signing; our model finds the words, '
                               'the pretrained model names them. Ours: our BiGRU-CTC alone.')
    if not hybrid_ready:
        st.caption('Sign by sign and Hybrid need model.tflite and sign_to_prediction_index_map.json '
                   'in models/kaggle_1st/.')
    all_signs = st.checkbox('Allow all 250 signs', False, disabled=not hybrid_ready,
                            help='Pretrained model only. Off: only our 52-sign vocabulary.')
    min_conf = st.slider('Ignore gestures below this confidence', 0.0, 0.9, 0.3, 0.05, disabled=not hybrid_ready,
                         help='Pretrained model only: a gesture that is not a known sign gets low confidence '
                              'and is dropped instead of shown as a random word.')
    if recogniser == SIGN_BY_SIGN:
        sign_gap_s = st.slider('Sign by sign: hands-down gap between signs (s)', 0.15, 1.0, 0.3, 0.05)
        sentence_end_s = st.slider('Sign by sign: hands-down time that ends the sentence (s)', 0.8, 4.0, 1.5, 0.1)
    with st.expander(f"Vocabulary ({len(bundle_base['vocab'])} ASL signs)"):
        st.write(', '.join(sorted(bundle_base['vocab'])))
    with st.expander('Demo sentences'):
        st.write('\n'.join(f'- {s}' for s in DEMO_SENTENCES))
    st.caption('Frames are processed on this computer and never stored. '
               'This prototype makes mistakes and must not replace a qualified interpreter.')

bundle = dict(bundle_base, threshold=threshold)
vocab = None if all_signs else bundle['vocab']


def recognise(sentence):
    if recogniser == CONTINUOUS:
        return pretrained_isolated.predict_sliding(sentence, load_pretrained(), vocab, threshold, min_conf)
    if recogniser == SIGN_BY_SIGN:
        clips, starts = split_signs(sentence, gap_s=sign_gap_s, fps=target_fps)
        return pretrained_isolated.predict_signs(clips, load_pretrained(), vocab, threshold, min_conf, starts)
    if recogniser == HYBRID:
        return pretrained_isolated.predict_hybrid(sentence, bundle, load_pretrained(), vocab, min_confidence=min_conf)
    return sb.predict(sentence, bundle)


DEBUG_DIR = ROOT / 'debug'


def save_sentence(sentence, result, reference, fps):
    DEBUG_DIR.mkdir(exist_ok=True)
    name = time.strftime('%H%M%S') + '_' + (reference.replace(' ', '-') or 'live')
    np.savez_compressed(DEBUG_DIR / f'{name}.npz', landmarks=sentence, recognised=result['text'],
                        reference=reference, recogniser=recogniser, camera_fps=fps)


def camera_loop(on_sentence, key, reference=''):
    """Run the webcam until the toggle is switched off. Calls on_sentence(result, frames) per sentence."""
    if not st.toggle('Start camera', key=key):
        st.info('Switch on the camera, sign a sentence, then lower your hands for a moment to finish it.')
        return
    frame_ph, status_ph = st.empty(), st.empty()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        st.error(f'Could not open camera {cam_index}. Close other apps that use it, or try another index.')
        return
    end_s = sentence_end_s if recogniser == SIGN_BY_SIGN else pause_s
    segmenter = SentenceSegmenter(pause_s=end_s, target_fps=target_fps)
    extractor = HolisticExtractor()
    t0 = last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error('The camera stopped sending frames.')
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            t_ms = (now - t0) * 1000
            landmarks = extractor.process(frame, t_ms)
            sentence = segmenter.push(landmarks, t_ms)
            if sentence is not None:
                result = recognise(sentence)
                if save_debug:
                    save_sentence(sentence, result, reference, fps)
                on_sentence(result, len(sentence))
            view = draw_landmarks(frame, landmarks) if show_overlay else frame
            frame_ph.image(cv2.flip(view, 1), channels='BGR', width='stretch')   # mirrored like a selfie view
            hint = (f'lower your hands briefly between signs, and for {end_s:.1f} s to finish'
                    if recogniser == SIGN_BY_SIGN else 'lower your hands to finish')
            state = f'✋ signing… {hint}' if segmenter.active else 'waiting for hands'
            status_ph.caption(f'{state} · {fps:.0f} fps')
    finally:
        cap.release()
        extractor.close()


# ---------------------------------------------------------------- live webcam
if mode == 'Live webcam':
    st.header('Live sign-to-text')
    st.session_state.setdefault('history', [])
    cam_col, text_col = st.columns([3, 2])
    with text_col:
        st.subheader('Recognised sentences')
        if st.button('Clear'):
            st.session_state.history = []
        history_ph = st.empty()

    def show_history():
        items = st.session_state.history[-8:][::-1]
        body = ''.join(result_html(r, n) for r, n in items) or '<span style="opacity:.6">Nothing yet.</span>'
        history_ph.markdown(body, unsafe_allow_html=True)

    def on_sentence(result, frames):
        st.session_state.history.append((result, frames))
        show_history()

    show_history()
    with cam_col:
        camera_loop(on_sentence, key='cam_live')

# ---------------------------------------------------------------- uploaded video
elif mode == 'Upload video':
    st.header('Recognise a recorded video')
    upload = st.file_uploader('Video of signing', type=['mp4', 'mov', 'avi', 'webm', 'mkv'])
    whole = st.checkbox('Treat the whole video as one sentence', value=False,
                        help='Otherwise the video is split into sentences wherever the hands go down.')
    if upload is not None and st.button('Recognise', type='primary'):
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as f:
            f.write(upload.getbuffer())
            path = f.name
        try:
            bar = st.progress(0.0, text='Extracting landmarks…')
            landmarks, times = video_to_landmarks(path, progress=lambda p: bar.progress(p, text='Extracting landmarks…'))
        finally:
            os.unlink(path)   # the uploaded video is not kept
        bar.empty()
        if whole:
            sentences = [resample(landmarks, times, target_fps)] if len(landmarks) else []
        else:
            end_s = sentence_end_s if recogniser == SIGN_BY_SIGN else pause_s
            segmenter = SentenceSegmenter(pause_s=end_s, target_fps=target_fps)
            sentences = [s for s in (segmenter.push(lm, t) for lm, t in zip(landmarks, times)) if s is not None]
            tail = segmenter.flush()
            if tail is not None:
                sentences.append(tail)
        st.caption(f'{len(landmarks)} frames read, {len(sentences)} sentence(s) found.')
        if not sentences:
            st.warning('No signing found: no hands were detected long enough. Check lighting and framing.')
        for i, s in enumerate(sentences, 1):
            st.markdown(f'**Sentence {i}**' + result_html(recognise(s), len(s)), unsafe_allow_html=True)

# ---------------------------------------------------------------- live evaluation (Member 4)
else:
    st.header('Live evaluation')
    st.caption(f'Each recognised sentence is scored against the chosen reference and saved to '
               f'{LOG_PATH.relative_to(ROOT)}. PS-41 asks for at least 10 complete sentences.')
    ref_col, custom_col = st.columns(2)
    reference = ref_col.selectbox('Sentence to sign next', DEMO_SENTENCES)
    custom = custom_col.text_input('…or type another reference (signs from the vocabulary)')
    reference = (custom or reference).strip().lower()
    unknown = [w for w in reference.split() if w not in bundle['vocab']]
    if unknown:
        st.warning(f'Not in the vocabulary, so these can never be recognised: {", ".join(unknown)}')
    st.markdown(f'### Sign: `{reference}`')

    cam_col, res_col = st.columns([3, 2])
    with res_col:
        last_ph, metrics_ph, table_ph = st.empty(), st.empty(), st.empty()

    def show_results():
        df = load_results()
        s = summarize(df)
        if s['attempts']:
            metrics_ph.markdown(f"**Live WER {s['wer']:.1%}** · {s['exact_sentences']}/{s['attempts']} sentences exact · "
                                f"{s['distinct_sentences']} distinct · median {s['median_latency_ms']:.0f} ms")
            table_ph.dataframe(df[['reference', 'hypothesis', 'mean_confidence', 'latency_ms']].iloc[::-1],
                               hide_index=True)
        else:
            metrics_ph.caption('No attempts recorded yet.')
            table_ph.empty()

    def on_sentence(result, frames):
        append_result(reference, result, frames)
        last_ph.markdown(result_html(result, frames, reference), unsafe_allow_html=True)
        show_results()

    show_results()
    with res_col:
        df = load_results()
        if not df.empty:
            st.download_button('Download results CSV', df.to_csv(index=False), file_name='live_results.csv')
            if st.checkbox('Delete all recorded attempts'):
                if st.button('Confirm delete'):
                    LOG_PATH.unlink(missing_ok=True)
                    st.rerun()
    with cam_col:
        camera_loop(on_sentence, key='cam_eval', reference=reference)
